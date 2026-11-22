# TORGO severity labels

Copy speaker-level severity labels from the official TORGO documentation. Record
the exact citation in `source` and describe the documented scale and its ordinal
encoding in `scale`. Higher encoded values must represent greater documented
severity. Leave a speaker's value null when the official documentation supplies
no label. Never guess or infer severity labels from audio, model scores, or other
speaker information.

F04 is a locked test speaker. Keep F04 out of candidate selection and severity
probes until the final check. The checked-in template leaves every speaker label
null. Keep any populated speaker-level labels in an approved private location.
