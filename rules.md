# Berkeley + Any

Piece movement follows chess on the referee's board. Each player sees only
their own pieces and the referee announcements. The API lists attempts the
player may submit. Hidden blockers and attacks can cause the referee to refuse
an attempt. A refusal leaves the same player on move. Avoid repeating an
attempt already refused in this turn.

A capture announces its square. A check announces its direction relative to
the king, or that it is a knight check. The referee omits the enemy piece
identity and special-move details. Stalemate is a draw.

When available, **Any?** asks whether a legal pawn capture exists. A positive
answer requires a legal pawn capture to finish the turn. The referee announces
the answer to both players.

Sources [Kriegspiel rules](https://kriegspiel.org/rules/berkeley) and
[engine policies](https://github.com/Kriegspiel/ks-game).
