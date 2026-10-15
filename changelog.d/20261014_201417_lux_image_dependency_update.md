### Security

- \[Lux\] The server image bumps Django to 5.2.18, cryptography to 50.0.1,
  lxml to 6.1.3, sqlparse to 0.6.0, urllib3 to 2.8.0, idna to 3.15, pyasn1
  to 0.6.4 and Django REST framework to 3.17.2
- \[Lux\] The server image no longer carries python3-venv or the pip and
  setuptools wheels it bundles once the virtual environment is built
- \[Lux\] The UI image upgrades its Alpine packages at build time
