# Rulesets

The worker uses [variant guidance](rules.py) to interpret the referee's
announcements. The game API supplies the current permitted attempts and
available questions. The installed engine decides which attempts succeed.

- [Berkeley and Berkeley + Any](https://kriegspiel.org/rules/berkeley) use
  square-only capture announcements.
- [Cincinnati](https://kriegspiel.org/rules/cincinnati) announces whether a
  pawn capture exists.
- [Wild 16](https://kriegspiel.org/rules/wild16) gives a pawn-capture count
  and keeps rejected attempts private from the opponent.
- [RAND](https://kriegspiel.org/rules/rand) announces eligible pawn sources
  and treats stalemate as a loss for the side to move.
- [English](https://kriegspiel.org/rules/english) allows one pawn attempt
  after a positive Any? and explicitly announces en passant.
- [CrazyKrieg](https://kriegspiel.org/rules/crazykrieg) adds public reserves
  and piece drops to the hidden-board game.
