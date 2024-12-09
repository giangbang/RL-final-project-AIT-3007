import numpy as np


def get_rewards_before_engagement(reds_in_cell, blues_in_cell,
                                  reds_in_cell_ef, reds_in_cell_force,
                                  blues_in_cell_ef, blues_in_cell_force,
                                  env, rewards, infos):
    """
    call from engage
    Rewards based on local consolidation of force (magnification of red/blue force)
    """

    for red in reds_in_cell:
        if red.alive:

            if (reds_in_cell_force > blues_in_cell_force) and (reds_in_cell_ef > blues_in_cell_ef):
                rewards[red.id] += 2.0 / np.pi * \
                                   np.arctan(reds_in_cell_force / blues_in_cell_force) * \
                                   np.arctan(reds_in_cell_ef / blues_in_cell_ef)  # [0.5,1)

            else:
                # rewards[red.id] += -0.5  # For trial-0
                rewards[red.id] += 0.1  # For trial-1

            infos[str(red.pos) + ' ' + red.id]['raw_reward'] = np.round(rewards[red.id], 1)
            infos[str(red.pos) + ' ' + red.id]['reward'] = np.round(rewards[red.id], 1)

    return rewards, infos


def get_consolidation_of_force_rewards(env, rewards):
    """
    Called from step() in battlefield_strategy.py (not used)
    """
    beta = 0.2
    coef = 0.04
    reds = []

    for red in env.reds:
        if red.alive:
            reds.append(red)

    if len(reds) >= 2:
        for red_i in reds:

            r2 = 0.0
            for red_j in reds:
                r2 += (red_i.pos[0] - red_j.pos[0]) ** 2 + (red_i.pos[1] - red_j.pos[1]) ** 2

            r_i = np.sqrt(r2 / len(reds))  # rms

            if r_i > 0:
                rewards[red_i.id] += \
                    2.0 / np.pi * np.arctan(1.0 / (beta * r_i)) * coef  # (0,0.04)
            else:  # Consolidate to one force
                rewards[red_i.id] += 1.0 * coef  # 0.04

    elif len(reds) == 1:  # Only one alive
        for red in reds:
            rewards[red.id] += 1.0 * coef  # 0.04

    else:  # len(reds) == 0
        pass

    return rewards


def get_economy_of_force_rewards(env, rewards):
    """
    Called from step() in battlefield_strategy.py (not used)
    """
    beta = 0.1
    coef = 0.04

    reds = []
    blues = []
    blues_force = []

    for red in env.reds:
        if red.alive:
            reds.append(red)

    for blue in env.blues:
        if blue.alive:
            blues.append(blue)
            blues_force.append(blue.force)

    max_blue_id = np.argmax(blues_force)
    max_blue = blues[max_blue_id]

    for red in reds:
        r2 = (red.pos[0] - max_blue.pos[0]) ** 2 + (red.pos[1] - max_blue.pos[1]) ** 2
        r = np.sqrt(r2)

        if r > 0:
            rewards[red.id] += 2.0 / np.pi * np.arctan(1.0 / (beta * r)) * coef  # (0,0.04)
        else:
            rewards[red.id] += 1.0 * coef  # 0.04

    return rewards


def get_rewards_after_engagement(reds_in_cell, blues_in_cell, env, rewards, infos):
    """
    At cell(x,y), rewards based on the engagement result  (not used)
    - reds_in_cell, blues_in_cell: Alive agents list before engagement
    - blue.force: Blue agent force after angagement
    """
    blues_force = 0.0
    for blue in blues_in_cell:
        blues_force += blue.force

    if blues_force <= env.config.threshold * 1.001:
        for red in reds_in_cell:
            rewards[red.id] += 2.0

    return rewards, infos

def add_info_next_ef_and_force(infos, agent):
    """
    (efficiency x force) and force after one step of Lanchester
    """
    infos[str(agent.pos) + ' ' + agent.id]['next_ef'] = np.round(agent.ef, 2)
    infos[str(agent.pos) + ' ' + agent.id]['next_force'] = np.round(agent.force, 2)

    return infos

def engage_and_get_rewards(env, x1, y1, rewards, infos):
    """
    Call from step()
    (x1,y1) ~ locations of engagement
    After one-step Lanchester simulation, red.ef, red.force, red.effective_ef, red.effective_force,
    blue.ef, blue.force, blue.effective_ef, blue.effective_force are updated.
    """

    for x, y in zip(x1, y1):
        """ before engage """
        # collect reds and blues in the cell
        reds_in_cell = []
        blues_in_cell = []

        for red in env.reds:
            if red.alive and red.pos == [x, y]:
                reds_in_cell.append(red)

                infos = set_info_red(env, red, infos)

        for blue in env.blues:
            if blue.alive and blue.pos == [x, y]:
                blues_in_cell.append(blue)

                infos = set_info_blue(env, blue, infos)

        if len(reds_in_cell) == 0 or len(blues_in_cell) == 0:
            raise ValueError()

        # Compute rR and R in the cell before engage (for Lanchester simulations)
        (reds_in_cell_ef, reds_in_cell_force, reds_in_cell_effective_ef,
         reds_in_cell_effective_force) = compute_current_total_ef_and_force(reds_in_cell)

        # Compute bB and b in the cell before engage (for Lanchester simulations)
        (blues_in_cell_ef, blues_in_cell_force, blues_in_cell_effective_ef,
         blues_in_cell_effective_force) = compute_current_total_ef_and_force(blues_in_cell)

        """
        Reward for consolidation of force 
        - Compute rewards of cell(x,y) based on before-engagement reds & blues status 
        """
        rewards, infos = \
            get_rewards_before_engagement(reds_in_cell, blues_in_cell,
                                          reds_in_cell_ef, reds_in_cell_force,
                                          blues_in_cell_ef, blues_in_cell_force,
                                          env, rewards, infos)

        """ engage (1 step of Lanchester simulation """
        # Update force of reds & blues in the cell
        # R_i' = R_i * (1 - bB / R * dt)
        next_red_force = []
        for red in reds_in_cell:
            next_red_force.append(
                max(red.force * (1 - blues_in_cell_ef / reds_in_cell_force * env.config.dt),
                    env.config.threshold))

        # B_i' = B_i * (1 - rR / B * dt)
        next_blue_force = []
        for blue in blues_in_cell:
            next_blue_force.append(
                max(blue.force * (1 - reds_in_cell_ef / blues_in_cell_force * env.config.dt),
                    env.config.threshold))

        """ after engage """
        # Update force, ef of reds & blues in the cell
        for i, red in enumerate(reds_in_cell):
            red.force = next_red_force[i]
            red.ef = red.force * red.efficiency

            red.effective_force = red.force - red.threshold
            red.effective_ef = red.ef - red.threshold * red.efficiency

            infos = add_info_next_ef_and_force(infos, red)

        for j, blue in enumerate(blues_in_cell):
            blue.force = next_blue_force[j]
            blue.ef = blue.force * blue.efficiency

            blue.effective_force = blue.force - blue.threshold
            blue.effective_ef = blue.ef - blue.threshold * blue.efficiency

            infos = add_info_next_ef_and_force(infos, blue)

        """
        Rewards after engagement
        - Compute rewards of cell(x,y) based on after-engagement reds & blues status
        """
        # rewards, infos = \
        #     get_rewards_after_engagement(reds_in_cell, blues_in_cell, env, rewards, infos)

    return rewards, infos