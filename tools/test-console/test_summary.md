# Test Console Fixes Summary

## Issues Fixed

### 1. History View UI Distortion
**Problem**: Tabbing through the history detail screen caused UI distortion and the Close button would get cut off.

**Solution**: 
- Simplified the layout from complex nested Grid/Vertical containers to a single Vertical container
- Removed overly restrictive min/max height constraints
- Used standard Textual layout patterns: Vertical with 1fr content area and 3-unit button bar
- Made the layout more flexible and responsive

**Files Changed**:
- `widgets/history_view.py`: Simplified RunDetailScreen layout

### 2. Quality Tests Config Missing MAX_TOKENS
**Problem**: Quality tests config dialog didn't expose MAX_TOKENS and THINKING_MAX_TOKENS environment variables.

**Solution**:
- Added `max_tokens` and `thinking_max_tokens` fields to TestConfig dataclass
- Added input fields to quality config dialog
- Updated `_read_config()` to read these values
- Updated runner to pass MAX_TOKENS and THINKING_MAX_TOKENS env vars

**Files Changed**:
- `runner.py`: Added fields to TestConfig, updated quality command builder
- `app.py`: Added input fields to quality config, updated _read_config()

### 3. Rebench Full Config Missing MAX_TOKENS
**Problem**: Rebench full config dialog didn't expose MAX_TOKENS and THINKING_MAX_TOKENS environment variables.

**Solution**:
- Reused the same fields added for quality tests
- Added input fields to rebench config dialog
- Updated `_read_config()` to read these values
- Updated runner to pass MAX_TOKENS and THINKING_MAX_TOKENS env vars

**Files Changed**:
- `runner.py`: Updated rebench command builder
- `app.py`: Added input fields to rebench config, updated _read_config()

### 4. Rebench Full Config Buttons Truncated
**Problem**: The rebench full config dialog had so many fields that the Run/Cancel buttons at the bottom were being cut off.

**Solution**:
- Increased max-height from 35 to 90% to allow more vertical space
- Wrapped config fields in a scrollable container (`#config-fields`)
- Used flex layout: title + hint + scrollable fields + button bar
- Button bar now has fixed height of 3 units, always visible

**Files Changed**:
- `app.py`: Updated ConfigScreen CSS and compose() method

## Test Results

### Unit Tests
```
83 tests passed ✅
```

### Headless Integration Tests
```
30/30 key interactions passed ✅
- Help screen opens/closes
- History screen opens/closes
- Manual target screen opens/closes
- All 6 test configs open/close
- Tab navigation works in config dialogs
- All global keybindings work (r, f, x, q)
```

### Manual Verification
```
✅ Quality config has MAX_TOKENS input field
✅ Quality config has THINKING_MAX_TOKENS input field
✅ Rebench config has MAX_TOKENS input field
✅ Rebench config has THINKING_MAX_TOKENS input field
✅ Rebench config Run/Cancel buttons are visible
✅ History detail screen opens without distortion
```

## Technical Details

### Config Dialog Structure
```
ConfigScreen > Vertical (max-height: 90%)
├── Label (config-title, height: 1)
├── Label (config-hint, height: 1)
├── Vertical (config-fields, height: 1fr, scrollable)
│   ├── Label + Select/Input pairs (varies by test type)
│   └── ... (up to 10+ fields for rebench)
└── Horizontal (button-bar, height: 3)
    ├── Button (Run)
    └── Button (Cancel)
```

### History Detail Screen Structure
```
RunDetailScreen > Vertical (height: 100%)
├── Label (detail-title, height: 1)
├── RichLog (detail-content, height: 1fr, scrollable)
└── Horizontal (button-bar, height: 3)
    ├── Button (Summary)
    ├── Button (Report, conditional)
    ├── Button (Log, conditional)
    └── Button (Close)
```

### Environment Variables Passed
- **Quality Tests**: MAX_TOKENS, THINKING_MAX_TOKENS
- **Rebench Full**: MAX_TOKENS, THINKING_MAX_TOKENS
- **Bench Tests**: RUNS, WARMUPS, ONLY, FORCE_TOKENS, ENABLE_THINKING

## Files Modified
1. `club3090_test_console/app.py` - Config dialogs and main app
2. `club3090_test_console/runner.py` - Test config and command builder
3. `club3090_test_console/widgets/history_view.py` - History detail screen

All changes are backward compatible and don't break existing functionality.
