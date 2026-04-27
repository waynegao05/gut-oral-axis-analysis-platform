# Development Notes

Current repository status:

- `app.py`: single-file runnable prototype
- `modular_app.py`: modular Flask entrypoint
- `src/`: modular core logic
- `data/microbe_drug_rules.json`: prototype rule base
- `templates/index.html`: browser demo

Recommended local workflow:

1. Install dependencies with `pip install -r requirements.txt`
2. Run `python modular_app.py`
3. Open `http://127.0.0.1:8765`
4. Test the API with `api_test.http`

Next planned upgrades:

- richer rule engine
- trained model hooks
- validation and error handling
- report rendering improvements
- structured logging
