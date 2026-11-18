"""Referee guidance for the playable Kriegspiel.org rulesets."""

_REFEREE = """Players see their own pieces and the referee's announcements.
The list of available attempts reflects what may be tried. Only the referee
knows which attempts are legal on the hidden board. The turn continues after
a refusal. Checks indicate rank, file, long or short diagonal, or knight, with
multiple directions possible. A check announcement gives the kind of threat
while leaving the checking piece's square hidden."""

_BERKELEY = """Captures announce the captured square only. En passant uses
the captured pawn's square without identifying the special move. Castling and
promotion are silent. Stalemate is a draw."""

_ONE_PAWN_TRY = """When ask_any is available, it asks whether a legal pawn
capture exists. A positive answer requires one pawn-capture attempt. If the
referee rejects it as illegal, the mover can finish with another kind of move.
A negative answer removes pawn-capture attempts for the rest of the turn."""

_RULES = {
    "berkeley": f"""Berkeley
{_BERKELEY}
There is no Ask Any or advance pawn-capture announcement.""",
    "berkeley_any": f"""Berkeley + Any
{_BERKELEY}
When ask_any is available, it asks whether a legal pawn capture exists. A
positive answer requires a successful pawn capture to finish the turn. A
negative answer removes pawn-capture attempts for the rest of the turn.""",
    "cincinnati": """Cincinnati
The turn announcement says whether any legal pawn capture exists. A positive
announcement allows pawn-capture attempts without obligating a pawn capture.
Without it, a pawn-capture attempt can be Nonsense. There is no Ask Any.
Captures announce the removed square and whether the captured unit was a
pawn or a non-pawn piece. Castling and promotion are silent. Stalemate is a draw.""",
    "wild16": """Wild 16
The turn announcement gives the count of legal pawn captures. Promotions to
different pieces on the same capture count as one try. The count does not
identify the pawns or targets and does not require the player to capture.
There is no Ask Any. A refused attempt is private to its mover.
Captures announce the removed square and pawn versus non-pawn identity. En
passant uses the removed pawn's square. Promotion and castling are silent.
Stalemate is a draw.""",
    "rand": """RAND
The turn announcement gives the distinct source squares of pawns with legal
captures. Targets remain hidden, and a legal capture must resolve any check.
These notices permit pawn captures without requiring one. There is no Ask Any.
Captures announce the removed square and pawn versus non-pawn identity. The
referee announces that a promotion occurred while keeping its square and
resulting piece type hidden. Castling is silent. The stalemated player loses.""",
    "english": f"""English
Captures announce the square without identifying the captured unit. En
passant is announced explicitly at the capturing pawn's destination. Castling
and promotion are silent. Stalemate is a draw.
{_ONE_PAWN_TRY}""",
    "crazykrieg": f"""CrazyKrieg
Captured units change sides and enter the capturer's reserve. Both reserves
are public. A captured promoted pawn returns to reserve as a pawn. Captures
announce the removed square and the unit's exact reserve type.
An available drop such as N@f3 places a reserved piece on an empty square and
uses that piece from the reserve. A drop cannot capture. Pawn drops must be on
ranks two through seven. A drop must leave the mover's king safe and may block
a check or deliver check or mate. A successful drop reveals its type while
keeping its square hidden from the opponent. Promotion and castling are
silent. Stalemate is a draw.
{_ONE_PAWN_TRY}""",
}

SUPPORTED_VARIANTS = tuple(_RULES)


def rules_for(variant: str) -> str:
    """Return the rules for a supported variant, rejecting unknown variants."""
    return f"{_RULES[variant]}\n\n{_REFEREE}"
