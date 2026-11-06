### Security

- \[Lux\] The server image builds smokescreen with Go 1.27.3 and
  golang.org/x/net 0.60.0, takes Ubuntu's pending security updates
  (libpcre2-8-0 10.42-4ubuntu2.2), and moves to django-allauth 65.19.7 with
  oauthlib 4.0.1, pyasn1 0.6.5, and idna 3.20. Ubuntu's venv and pip wheel
  packages, whose fixes ship only in Ubuntu Pro, leave the final image, as do
  the requirement locks, which scanners read as installed packages. The DICOM
  converter moves to pydicom 3.0.2.
- \[Lux\] The server image no longer installs 7zip, whose CVE-2023-52168 and
  CVE-2023-52169 fixes ship only in Ubuntu Pro. Zip, tar, and rar uploads still
  extract; `.7z` archives are refused.
- \[Lux\] The YOLOv7 sidecar takes Ubuntu's pending security updates (pcre2,
  libx11, zlib) and drops Ubuntu's venv and pip wheel packages once its
  virtualenv is built. The UI image upgrades its Alpine packages, which moves
  zlib to 1.3.2-r1.
