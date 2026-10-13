### Security

- \[Lux\] The server image upgrades its Ubuntu packages at build time, builds
  smokescreen with Go 1.27.2 and current x/net, x/text and logrus, and bumps
  PyJWT to 2.15.1, NLTK to 3.10.3, AnyIO to 4.14.2 and fontTools to 4.66.1
- \[Lux\] The server image no longer carries the requirements manifests that
  production does not install, or the standalone DICOM converter
- \[Lux\] The UI image is served by nginx-unprivileged 1.31.6 on Alpine 3.24,
  and its Node build stage is pinned by digest
