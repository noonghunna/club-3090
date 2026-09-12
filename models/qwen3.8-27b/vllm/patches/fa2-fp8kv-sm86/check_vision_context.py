"""Exercise maximum image processing with a nearly full text context."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import base64
import json
import math
import time
import urllib.error
import urllib.request


@dataclass(frozen=True)
class Endpoint:
    address: str
    model: str

    def request(self, path: str, body: dict, timeout: int = 900) -> dict:
        request = urllib.request.Request(
            self.address + path, json.dumps(dict(body, model=self.model)).encode('utf-8'),
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'HTTP {error.code}: {error.read().decode("utf-8")[:1000]}') from error


@dataclass(frozen=True)
class VisionDocument:
    image: Path
    stamp: str
    minimum_pixels: int

    def payload(self, blocks: int) -> dict:
        block = (
            'This section describes the history of computing in detail. '
            'Transistors were invented in 1947 at Bell Labs. The integrated circuit came a decade later. '
            'Microprocessors emerged in the 1970s and changed the world. '
            'Personal computing followed, then networking, then the web, then cloud and AI. '
        )
        content = [
            {'type': 'text', 'text': f'Document {self.stamp}.\n' + block * (blocks // 2)
             + '\nThe document key is cobalt-orchid-5831.\n' + block * (blocks - blocks // 2)},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'
             + base64.b64encode(self.image.read_bytes()).decode('ascii')}},
            {'type': 'text', 'text': 'State the document key. List every shape in the image with its '
             'color and read the number. Be literal and brief.'},
        ]
        return {
            'messages': [{'role': 'user', 'content': content}],
            'chat_template_kwargs': {'enable_thinking': False, 'reasoning_effort': 'low'},
            # Ask the model's own processor to exercise the configured 4 MP ceiling.
            # The source fixture remains unchanged; it is 640x480, not a native 4 MP file.
            # Leave room for patch-grid rounding so the processed image stays below 4096 tokens.
            'mm_processor_kwargs': {'size': {'shortest_edge': self.minimum_pixels, 'longest_edge': 4194304}},
        }

    def sized(self, endpoint: Endpoint, target: int) -> tuple[dict, int]:
        base = endpoint.request('/tokenize', self.payload(0), timeout=120)['count']
        hundred = endpoint.request('/tokenize', self.payload(100), timeout=120)['count']
        per_block = (hundred - base) / 100
        if per_block <= 0:
            raise RuntimeError('Tokenizer calibration did not grow with the document')
        blocks = max(0, math.floor((target - base) / per_block))
        for attempt in range(3):
            payload = self.payload(blocks)
            count = endpoint.request('/tokenize', payload, timeout=120)['count']
            if target - 512 <= count <= target:
                return payload, count
            blocks = max(0, blocks + math.floor((target - count) / per_block))
        raise RuntimeError(f'Unexpected calibrated context length: {count}, target {target}')


@dataclass(frozen=True)
class MeasuredRequest:
    endpoint: Endpoint
    label: str
    payload: dict
    minimum_tokens: int

    def run(self) -> dict:
        print(json.dumps({'phase': self.label, 'event': 'start',
                          'utc': datetime.now(timezone.utc).isoformat()}), flush=True)
        started = time.monotonic()
        result = self.endpoint.request('/v1/chat/completions', dict(
            self.payload, max_tokens=256, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0))
        elapsed = time.monotonic() - started
        message = result['choices'][0]['message']
        text = message.get('content') or ''
        lower = text.lower()
        facts = {
            'document_key': 'cobalt-orchid-5831' in lower,
            'red_circle': 'red' in lower and ('circle' in lower or 'ellipse' in lower),
            'blue_square': 'blue' in lower and ('square' in lower or 'rectangle' in lower),
            'green_triangle': 'green' in lower and 'triangle' in lower,
            'number_47': '47' in lower,
        }
        tokens = result['usage']['prompt_tokens']
        passed = all(facts.values()) and self.minimum_tokens <= tokens < 261888
        record = {'phase': self.label, 'event': 'end',
                  'utc': datetime.now(timezone.utc).isoformat(), 'seconds': round(elapsed, 3),
                  'usage': result['usage'], 'content': text, 'facts': facts, 'passed': passed}
        print(json.dumps(record), flush=True)
        if not passed:
            raise RuntimeError(f'{self.label} failed its facts or context-size checks')
        return message


parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8144')
parser.add_argument('--model', default='qwen3.8-27b')
parser.add_argument('--target', type=int, default=260000)
parser.add_argument('--calibration-only', action='store_true')
parser.add_argument('--image-only', action='store_true')
parser.add_argument('--image-min-pixels', type=int, choices=[65536, 1048576, 2097152, 4100000], default=4100000)
args = parser.parse_args()
if not 10000 <= args.target <= 260000:
    parser.error('--target must be between 10000 and 260000')
endpoint = Endpoint(args.url, args.model)
document = VisionDocument(Path('scripts/assets/vision-test.png'), datetime.now(timezone.utc).isoformat(), args.image_min_pixels)
small = document.payload(0)
small_tokenization = endpoint.request('/tokenize', dict(small, return_token_strs=True), timeout=120)
small_tokens = small_tokenization['count']
image_tokens = small_tokenization['token_strs'].count('<|image_pad|>')
print(json.dumps({'phase': 'image-calibration', 'tokens': small_tokens,
                  'image_tokens': image_tokens,
                  'requested_min_pixels': args.image_min_pixels,
                  'source_pixels': [640, 480], 'processor_pixel_budget': 4194304}), flush=True)
if not args.image_min_pixels / 1024 * 0.8 <= image_tokens <= 4096:
    raise RuntimeError('Image request did not fit the configured large image-token budget')
if args.calibration_only:
    raise SystemExit(0)
MeasuredRequest(endpoint, 'image-processing', small, image_tokens).run()
if args.image_only:
    raise SystemExit(0)
payload, count = document.sized(endpoint, args.target)
print(json.dumps({'phase': 'context-calibration', 'tokens': count}), flush=True)
answer = MeasuredRequest(endpoint, 'long-context-image', payload, args.target - 512).run()
for turn in range(1, 3):
    payload = dict(payload, messages=[*payload['messages'], answer,
        {'role': 'user', 'content': 'Repeat the document key, the three colored shapes, and the number.'}])
    answer = MeasuredRequest(endpoint, f'long-context-followup-{turn}', payload, args.target - 512).run()
