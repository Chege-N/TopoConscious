# Contributing to TopoConscious

Bug reports, feature requests, and pull requests are welcome.

## Development setup
git clone https://github.com/Chege-N/TopoConscious.git
cd TopoConscious
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest tests/ -v

## Reporting issues
Open a GitHub issue with a minimal reproducible example.

## Code style
Follow PEP 8. Max line length 100. Run flake8 before submitting a PR.
