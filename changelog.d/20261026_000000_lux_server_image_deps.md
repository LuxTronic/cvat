### Security

- \[Lux\] The server image upgrades zlib1g to the Ubuntu 24.04 security
  revision, pins fonttools 4.67.0, nltk 3.10.4, and tornado 6.6, and builds
  smokescreen from current Stripe source with Go 1.27.2. Smokescreen starts
  with `--allow-self-connections`, and its read and write deadlines are 24
  hours. An allow rule names the storage port, including when that address
  is on the proxy host, and a long import keeps working.
