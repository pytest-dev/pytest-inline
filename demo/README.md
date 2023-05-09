# pytest-inline demo

Run all demos
```
pytest
```

Run an example basic inline test (try: uncomment the failing inline test in example.py and see how the output changes):
```
pytest example.py
```

Explore features supported by pytest-inline in features.py
```
pytest -vv features.py
pytest -vv features.py --inlinetest-group=foo
pytest -vv features.py --inlinetest-order=bar  --inlinetest-order=foo
```

Run inline tests in parallel
```
pip install pytest-xdist
pytest parallel/          # sequential
pytest -n 4 parallel/  # parallel using 4 threads
```

