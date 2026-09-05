"""Regenerate the unnumbered-policy test fixture.

The fixture exercises the segmentation path the baseline regex cannot handle:
prose headings with no numbering. It is synthetic, so its headings are cleanly
larger and bold by construction -- weaker evidence than a real document. Drop a
real privacy policy or ToS into data/raw/ and re-run the tests when you can.

    python scripts/make_policy_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf  # noqa: E402

from src.config import PATHS  # noqa: E402

SECTIONS = [
    ("How We Collect Your Information",
     "We collect information you provide directly to us when you create an account, "
     "including your name, email address and payment details. We also automatically "
     "collect device identifiers, IP address and browsing activity through cookies."),
    ("Sharing With Third Parties",
     "We may share your personal data with advertising partners, analytics providers "
     "and payment processors. We do not sell your personal information to third "
     "parties for monetary consideration."),
    ("How Long We Keep Your Data",
     "We retain your account information for as long as your account remains active "
     "and for a further twenty-four months after closure, unless a longer period is "
     "required by applicable law."),
    ("Your Rights and Choices",
     "You may request access to, correction of, or deletion of your personal data at "
     "any time by contacting our privacy team. You may also opt out of marketing "
     "communications using the unsubscribe link."),
    ("Automatic Renewal of Your Subscription",
     "Your subscription will automatically renew at the end of each billing period "
     "unless you cancel at least fourteen days before the renewal date. Renewal "
     "charges are applied to the payment method on file."),
    ("Resolving Disputes",
     "Any dispute arising under these terms shall be resolved by binding arbitration "
     "administered in the State of Delaware. You waive the right to participate in a "
     "class action."),
]


def main() -> int:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60
    page.insert_text((60, y), "Sample Privacy Policy and Terms of Service",
                     fontname="hebo", fontsize=18)
    y += 40
    for title, body in SECTIONS:
        if y > 700:
            page = doc.new_page()
            y = 60
        page.insert_text((60, y), title, fontname="hebo", fontsize=14)
        y += 22
        for chunk in (body[i:i + 95] for i in range(0, len(body), 95)):
            page.insert_text((60, y), chunk, fontname="helv", fontsize=11)
            y += 15
        y += 18

    out = PATHS.data_raw / "synthetic-unnumbered-policy.pdf"
    doc.save(out)
    doc.close()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
