"""
test.py -c BBRI: Menampilan data finansial dari perusahaan BBRI, default = BBRI
test.py -s peers: menampilkan data perusahaan sejenis dari perusahaan BBRI, default = financials
test.py --nice: menampilkan data dengan format yang lebih rapi
"""

import os
from dotenv import load_dotenv
import requests
import argparse as ag


# Pakai _shoot soalnya private function, gak bisa diakses dari luar
def _shoot(url):
    """
    Request handler to Sectors Website
    """

    SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")
    headers = {"Authorization": SECTORS_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)


def sectors(company="BBRI", section="dividend"):
    """
    URL handler to https://sectors.app/
    :company: fiils with company that available in the documentation
    :section: consist of overview, valuation, future, peers, financial, dividend, management, ownership
    """

    valid_section = [
        "overview",
        "valuation",
        "future",
        "peers",
        "financials",
        "dividend",
        "management",
        "ownership",
    ]
    assert (
        section in valid_section
    ), "No no NO! You need to specify what report you want to extract! Check the documentation."
    assert (
        len(company) == 4
    ), "What? That company doesn't exist gurrrrrrl. Must have 4 digit code of the company to get inside. xoxo."

    return _shoot(
        f"https://api.sectors.app/v1/company/report/{company}/?sections={section}"
    )


# print(sectors("BBCA", "peers"))
# print(sectors.__doc__)

if __name__ == "__main__":
    parser = ag.ArgumentParser(
        description="Hear Ye! Hear Ye! Get your company reports here!"
    )

    # Add the arguments to the parser, useh -h or --help to see the description
    parser.add_argument(
        "-c", "--company", type=str, help="Company code", default="BBRI"
    )
    parser.add_argument(
        "-s",
        "--section",
        type=str,
        help="What report you want to extract?",
        default="financials",
    )
    parser.add_argument(
        "--nice",
        action="store_true",
        default=True,
        help="Uweee I cant read, I need to see the structure clearly :( ",
    )

    # call argparse to parse the arguments with `python test.py -c BBCA -s peers`
    args = parser.parse_args()
    load_dotenv()

    # print either nicer or not
    if args.nice:
        import pprint as pp

        pp.pprint(sectors(args.company, args.section))
    else:
        print(sectors(args.company, args.section))
