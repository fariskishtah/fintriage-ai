"""Synthetic example complaints, written for this project.

These are NOT real consumer narratives. No row from the dataset is exposed in
the UI; every example below was written by hand to exercise one class.
"""

EXAMPLES: list[tuple[str, str]] = [
    (
        "Credit Card",
        "I was charged an annual fee on my card after I had already called to "
        "close the account. The representative confirmed the closure on the phone "
        "but the fee still appeared on my next statement, and now interest is "
        "accruing on a balance I should not have.",
    ),
    (
        "Credit Reporting",
        "There is an account on my credit report that does not belong to me. I "
        "filed a dispute with the bureau over two months ago and submitted a "
        "police report, but the item is still listed as verified and my score "
        "has dropped more than eighty points.",
    ),
    (
        "Debt Collection",
        "A collection agency keeps calling me five or six times a day, including "
        "at my workplace after I asked them to stop. They are demanding payment "
        "for a debt I already settled, and they refuse to send me written "
        "validation of the amount they claim I owe.",
    ),
    (
        "Mortgages and Loans",
        "My mortgage servicer changed without any notice to me and my escrow "
        "payment jumped by four hundred dollars. When I asked for a breakdown of "
        "the new escrow analysis they sent an unreadable form, and my modification "
        "application has been pending for six months with no decision.",
    ),
    (
        "Retail Banking",
        "My checking account was overdrawn by a small amount and the bank charged "
        "me four separate overdraft fees in one day by reordering the transactions. "
        "I also could not access my funds for three days because they placed an "
        "unexplained hold on a payroll deposit.",
    ),
    (
        "Ambiguous (expect low confidence)",
        "A collector reported this account to the credit bureaus and now my "
        "mortgage application was denied because of it, so I am not sure who to "
        "complain about here.",
    ),
]


def example_texts() -> list[list[str]]:
    """Gradio's gr.Examples wants a list of input-rows."""
    return [[text] for _, text in EXAMPLES]
