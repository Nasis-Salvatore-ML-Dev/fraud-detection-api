## Activate environment

- python3 -m venv ~/.aerospace-defect-detection
- source ~/.aerospace-defect-detection/bin/activate

## Delete the environment

- rm -rf ~/.mlops-ch1-intro

Command Part Translation Why?

- python "Hey Python program!"
- -m "with the module..."
- pytest "...called pytest"
- -vv "and tell me EVERYTHING" v = verbose, vv = EXTRA verbose
- --cov=hello "and show me which lines of hello.py did we provide "coverage"
- test_hello.py "on this test file right here"

## Testing

testpaths = ["tests"] tells pytest where to look. addopts = "--cov=src --cov-report=term-missing" means every test run automatically produces a coverage report showing which lines in src/ are not covered.
