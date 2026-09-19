from arena.entities import Action
from arena.env import ArenaEnv


class RuleAgent:
    name = "rule"

    def act(self, env: ArenaEnv) -> Action:
        legal = env.legal_actions()
        if Action.HEAL in legal and env.player.hp <= 40:
            return Action.HEAL

        attacks = [action for action in legal if action.value.startswith("attack_")]
        if attacks and env.player.hp > 25:
            return attacks[0]

        moves = [action for action in legal if action.value.startswith("move_")]
        safe = [action for action in moves if env.add(env.player.position, action.value[-1]) not in env.fires]
        options = safe or moves
        if not options:
            return Action.WAIT

        targets = list(env.medkits) if env.player.hp <= 50 and env.medkits else list(env.gems)
        if not targets:
            targets = [enemy.position for enemy in env.enemies]

        def value(action: Action) -> tuple[int, int]:
            position = env.add(env.player.position, action.value[-1])
            distance = min((env._distance(position, target) for target in targets), default=0)
            backtrack = position == env.previous_player_position
            return distance, backtrack

        chosen = min(options, key=value)
        return chosen
