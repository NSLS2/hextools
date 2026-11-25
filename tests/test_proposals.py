import pytest

from hextools.utils import initialize_run_engine
from hextools.utils.proposals import MOCK_PROPOSAL, MockProposalClient, Proposal


def test_mock_client_offers_the_mock_startup_proposal(monkeypatch):
    monkeypatch.setenv("HEXTOOLS_RUNNING_IN_CI", "yes")
    md = initialize_run_engine().md
    client = MockProposalClient()

    proposal = client.get_proposal(MOCK_PROPOSAL.number)

    assert md["data_session"] == f"pass-{proposal.number}"
    assert md["proposal"] == {
        "title": proposal.title,
        "type": proposal.type,
        "pi_name": proposal.pi_name,
    }
    assert proposal in client.list_proposals().commissioning


def test_mock_client_generates_10_to_20_proposals_of_each_kind():
    listing = MockProposalClient().list_proposals()

    assert 10 <= len(listing.user) <= 20
    # The mock startup proposal is listed besides the generated ones.
    assert 11 <= len(listing.commissioning) <= 21
    numbers = [p.number for p in listing.user + listing.commissioning]
    assert len(set(numbers)) == len(numbers)
    for proposals in (listing.user, listing.commissioning):
        assert [int(p.number) for p in proposals] == sorted(
            (int(p.number) for p in proposals), reverse=True
        )


def test_mock_client_proposals_are_the_same_for_the_same_seed():
    assert MockProposalClient(3).list_proposals() == (
        MockProposalClient(3).list_proposals()
    )


def test_mock_client_looks_up_listed_proposals_and_rejects_unknown_ones():
    client = MockProposalClient()
    (listed, *_) = client.list_proposals().user

    assert client.get_proposal(int(listed.number)) == listed
    assert isinstance(listed, Proposal) and listed.title and listed.pi_name
    try:
        client.get_proposal("1")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown proposal was not rejected")


def test_mock_client_returns_the_metadata_of_the_new_proposal():
    client = MockProposalClient()
    (target, *_) = client.list_proposals().user

    md = client.change_proposal(target.number, "someone", "any password")

    assert md["data_session"] == f"pass-{target.number}"
    assert md["username"] == "someone"
    assert md["proposal"] == {
        "proposal_id": target.number,
        "title": target.title,
        "type": target.type,
        "pi_name": target.pi_name,
    }
    with pytest.raises(ValueError):
        client.change_proposal("1", "someone", "")
