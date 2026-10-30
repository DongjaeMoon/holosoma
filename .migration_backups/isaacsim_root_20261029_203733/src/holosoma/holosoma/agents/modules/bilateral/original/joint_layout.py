"""Joint-list statements copied from domi make_obs_cfg / make_act_cfg."""

def observation_joint_names(side, include_else=True):
    cont = "right" if side == "left" else "left"
    joint_names = [
        # waist
        'waist_yaw_joint',
        'waist_roll_joint',
        'waist_pitch_joint',

        # leg
        F'{side}_ankle_roll_joint',
        F'{side}_ankle_pitch_joint',
        F'{side}_hip_pitch_joint',
        F'{side}_hip_roll_joint',
        F'{side}_hip_yaw_joint',
        F'{side}_knee_joint',

        # arm
        F'{side}_shoulder_pitch_joint',
        F'{side}_shoulder_roll_joint',
        F'{side}_shoulder_yaw_joint',
        F'{side}_elbow_joint',
        F'{side}_wrist_roll_joint',
        F'{side}_wrist_pitch_joint',
        F'{side}_wrist_yaw_joint',
    ]

    if include_else:
        # include the "flip-side" joints for subnet, in multi-agent scenarios.
        # cont stand for "contralateral side"
        k_cont = [
            F'{cont}_ankle_roll_joint',
            F'{cont}_ankle_pitch_joint',
            F'{cont}_hip_pitch_joint',
            F'{cont}_hip_roll_joint',
            F'{cont}_hip_yaw_joint',
            F'{cont}_knee_joint',
            F'{cont}_shoulder_pitch_joint',
            F'{cont}_shoulder_roll_joint',
            F'{cont}_shoulder_yaw_joint',
            F'{cont}_elbow_joint',
            F'{cont}_wrist_roll_joint',
            F'{cont}_wrist_pitch_joint',
            F'{cont}_wrist_yaw_joint',
        ]
        joint_names.extend(k_cont)

    return joint_names

def action_joint_names(side, arm=True, skip_waist=False):
    waist_names = [
        'waist_yaw_joint',
        'waist_roll_joint',
        'waist_pitch_joint',
    ]

    leg_names = [
        # leg
        F"{side}_hip_pitch_joint",
        F"{side}_hip_roll_joint",
        F"{side}_hip_yaw_joint",
        F"{side}_knee_joint",
        F"{side}_ankle_pitch_joint",
        F"{side}_ankle_roll_joint",
    ]

    if skip_waist:
        joint_names = leg_names
    else:
        joint_names = [*waist_names, *leg_names]

    if arm:
        joint_names.extend([
            F'{side}_shoulder_pitch_joint',
            F'{side}_shoulder_roll_joint',
            F'{side}_shoulder_yaw_joint',
            F'{side}_elbow_joint',
            F'{side}_wrist_roll_joint',
            F'{side}_wrist_pitch_joint',
            F'{side}_wrist_yaw_joint'
        ])
    return joint_names
