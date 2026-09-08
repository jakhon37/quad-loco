"""Convert the original ROS URDF into a MuJoCo scene that can train on CPU."""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from quad_loco.constants import (
    ACTUATOR_FORCE,
    ACTUATOR_KP,
    ACTUATOR_KV,
    DEFAULT_JOINT_POS,
    FOOT_GEOM_NAMES,
    HOME_HEIGHT,
    JOINT_LIMITS,
    JOINT_NAMES,
    PHYSICS_DT,
)
from quad_loco.paths import (
    cleaned_urdf,
    mesh_dir,
    mjcf_dir,
    original_urdf,
    robot_xml,
    scene_xml,
)

MESH_MAP = {
    "base/base_link": "base_link.stl",
    "base/base_link_old": "base_link.stl",
    "leg_joint/leg_joint": "leg_joint.stl",
    "left_thigh/left_thigh": "left_thigh.stl",
    "right_thigh/right_thigh": "right_thigh.stl",
    "shin/shin": "shin.stl",
    "shin/shin_old": "shin.stl",
    "foot/foot": "foot.stl",
}

COLLISION_GEOMS = {
    "trunk": [
        {"name": "trunk_col", "type": "box", "size": "0.30 0.09 0.07", "pos": "0 0 0.02"},
    ],
    "FL_hip": [{"name": "FL_hip_col", "type": "sphere", "size": "0.05", "pos": "0 0 -0.03"}],
    "FR_hip": [{"name": "FR_hip_col", "type": "sphere", "size": "0.05", "pos": "0 0 -0.03"}],
    "RL_hip": [{"name": "RL_hip_col", "type": "sphere", "size": "0.05", "pos": "0 0 -0.03"}],
    "RR_hip": [{"name": "RR_hip_col", "type": "sphere", "size": "0.05", "pos": "0 0 -0.03"}],
    "FL_thigh": [{"name": "FL_thigh_col", "type": "capsule", "size": "0.035", "fromto": "0.02 0 0 0.40 0 0"}],
    "FR_thigh": [{"name": "FR_thigh_col", "type": "capsule", "size": "0.035", "fromto": "0.02 0 0 0.40 0 0"}],
    "RL_thigh": [{"name": "RL_thigh_col", "type": "capsule", "size": "0.035", "fromto": "0.02 0 0 0.40 0 0"}],
    "RR_thigh": [{"name": "RR_thigh_col", "type": "capsule", "size": "0.035", "fromto": "0.02 0 0 0.40 0 0"}],
    "FL_calf": [{"name": "FL_calf_col", "type": "capsule", "size": "0.025", "fromto": "0 0 0 -0.30 0 0"}],
    "FR_calf": [{"name": "FR_calf_col", "type": "capsule", "size": "0.025", "fromto": "0 0 0 -0.30 0 0"}],
    "RL_calf": [{"name": "RL_calf_col", "type": "capsule", "size": "0.025", "fromto": "0 0 0 -0.30 0 0"}],
    "RR_calf": [{"name": "RR_calf_col", "type": "capsule", "size": "0.025", "fromto": "0 0 0 -0.30 0 0"}],
    "FL_foot": [{"name": "FL_foot_col", "type": "sphere", "size": "0.035", "priority": "1"}],
    "FR_foot": [{"name": "FR_foot_col", "type": "sphere", "size": "0.035", "priority": "1"}],
    "RL_foot": [{"name": "RL_foot_col", "type": "sphere", "size": "0.035", "priority": "1"}],
    "RR_foot": [{"name": "RR_foot_col", "type": "sphere", "size": "0.035", "priority": "1"}],
}


def _map_mesh_filename(path: str) -> str:
    cleaned = path.replace("package://robot_description/meshes/", "")
    cleaned = re.sub(r"^(DAE|STL)/", "", cleaned)
    cleaned = re.sub(r"\.(dae|stl)$", "", cleaned, flags=re.IGNORECASE)
    if cleaned not in MESH_MAP:
        raise KeyError(f"Unmapped mesh path: {path}")
    return MESH_MAP[cleaned]


def _joint_family(name: str) -> str:
    if "hip" in name:
        return "hip"
    if "thigh" in name:
        return "thigh"
    if "calf" in name:
        return "calf"
    return "hip"


def write_cleaned_urdf(src: Path, dst: Path) -> Path:
    text = src.read_text()

    def repl(match: re.Match[str]) -> str:
        return _map_mesh_filename(match.group(0))

    text = re.sub(r"package://robot_description/meshes/[^\"']+", repl, text)

    for name in JOINT_NAMES:
        family = _joint_family(name)
        lo, hi = JOINT_LIMITS[family]
        pattern = rf'(<joint name="{name}"[\s\S]*?<limit )([^>]*)(/>)'

        def limit_repl(match: re.Match[str], lo=lo, hi=hi) -> str:
            return (
                f'{match.group(1)}effort="{ACTUATOR_FORCE:.1f}" velocity="21" '
                f'lower="{lo}" upper="{hi}"{match.group(3)}'
            )

        text = re.sub(pattern, limit_repl, text, count=1)

    insert = """
  <mujoco>
    <compiler meshdir="../meshes" balanceinertia="true" discardvisual="false" fusestatic="false"/>
  </mujoco>
  <link name="world"/>
  <joint name="floating_base" type="floating">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="world"/>
    <child link="trunk"/>
  </joint>
"""
    text = text.replace("<robot name=\"quadruped_robot\">", "<robot name=\"quadruped_robot\">" + insert, 1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
    return dst


def _compile_urdf(urdf_path: Path, out_xml: Path) -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(out_xml), model)


def _indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def _find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None


def _disable_mesh_collision(root: ET.Element) -> None:
    for geom in root.iter("geom"):
        mesh = geom.get("mesh")
        if mesh:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("group", "2")
            geom.set("density", "0")


def _add_collision_geoms(root: ET.Element) -> None:
    missing = []
    for body_name, geoms in COLLISION_GEOMS.items():
        body = _find_body(root, body_name)
        if body is None and body_name.endswith("_foot"):
            body = _find_body(root, body_name.replace("_foot", "_calf"))
            if body is not None:
                for spec in geoms:
                    spec = dict(spec)
                    spec["pos"] = spec.get("pos", "-0.345 0 0")
                    geoms_override = [spec]
                geoms = geoms_override
        if body is None:
            missing.append(body_name)
            continue
        for spec in geoms:
            geom = ET.SubElement(body, "geom")
            geom.set("class", "collision")
            for key, value in spec.items():
                geom.set(key, value)
            if "foot_col" in spec.get("name", ""):
                geom.set("friction", "1.0 0.02 0.01")
                geom.set("condim", "3")
                geom.set("solimp", "0.015 1 0.031")
    if missing:
        names = [b.get("name") for b in root.iter("body") if b.get("name")]
        raise RuntimeError(f"Missing bodies {missing}. Compiled bodies: {names}")


def _add_sites_and_camera(root: ET.Element) -> None:
    trunk = _find_body(root, "trunk")
    if trunk is not None:
        imu = ET.SubElement(trunk, "site")
        imu.set("name", "imu")
        imu.set("pos", "0 0 0")
        imu.set("size", "0.02")
        imu.set("group", "5")
        cam = ET.SubElement(trunk, "camera")
        cam.set("name", "track")
        cam.set("mode", "trackcom")
        cam.set("pos", "2.4 -2.0 1.3")
        cam.set("xyaxes", "1 1 0 -0.2 0.2 1")
    for foot in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
        body = _find_body(root, foot)
        if body is None:
            continue
        site = ET.SubElement(body, "site")
        site.set("name", f"{foot}_site")
        site.set("size", "0.02")
        site.set("group", "5")


def _set_joint_dynamics(root: ET.Element) -> None:
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if name not in JOINT_NAMES:
            continue
        joint.set("damping", str(ACTUATOR_KV))
        joint.set("armature", "0.01")
        family = _joint_family(name)
        lo, hi = JOINT_LIMITS[family]
        joint.set("range", f"{lo} {hi}")


def _add_actuators_sensors_keyframe(root: ET.Element, model) -> None:
    import mujoco

    existing = root.find("actuator")
    if existing is not None:
        root.remove(existing)
    actuator = ET.SubElement(root, "actuator")
    for name in JOINT_NAMES:
        family = _joint_family(name)
        lo, hi = JOINT_LIMITS[family]
        pos = ET.SubElement(actuator, "position")
        pos.set("name", name)
        pos.set("joint", name)
        pos.set("kp", str(ACTUATOR_KP))
        pos.set("kv", str(ACTUATOR_KV))
        pos.set("forcerange", f"-{ACTUATOR_FORCE} {ACTUATOR_FORCE}")
        pos.set("ctrlrange", f"{lo} {hi}")

    existing = root.find("sensor")
    if existing is not None:
        root.remove(existing)
    sensor = ET.SubElement(root, "sensor")
    vel = ET.SubElement(sensor, "velocimeter")
    vel.set("name", "base_linvel")
    vel.set("site", "imu")
    gyro = ET.SubElement(sensor, "gyro")
    gyro.set("name", "base_angvel")
    gyro.set("site", "imu")

    qpos = np.zeros(model.nq)
    # freejoint: x y z quat
    qpos[0:3] = (0.0, 0.0, HOME_HEIGHT)
    qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    ctrl = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint {name} missing from compiled model")
        qpos[model.jnt_qposadr[jid]] = DEFAULT_JOINT_POS[name]
        ctrl.append(DEFAULT_JOINT_POS[name])

    existing = root.find("keyframe")
    if existing is not None:
        root.remove(existing)
    keyframe = ET.SubElement(root, "keyframe")
    key = ET.SubElement(keyframe, "key")
    key.set("name", "home")
    key.set("qpos", " ".join(f"{v:.6f}" for v in qpos))
    key.set("ctrl", " ".join(f"{v:.6f}" for v in ctrl))


def _ensure_option_defaults(root: ET.Element) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    compiler.set("meshdir", "../meshes")
    compiler.set("balanceinertia", "true")
    compiler.set("fusestatic", "false")

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", str(PHYSICS_DT))
    option.set("integrator", "implicitfast")
    option.set("cone", "elliptic")
    option.set("impratio", "100")

    default = root.find("default")
    if default is None:
        default = ET.Element("default")
        root.insert(2, default)
    collision = None
    for child in default.findall("default"):
        if child.get("class") == "collision":
            collision = child
    if collision is None:
        collision = ET.SubElement(default, "default")
        collision.set("class", "collision")
    geom = collision.find("geom")
    if geom is None:
        geom = ET.SubElement(collision, "geom")
    geom.set("group", "3")
    geom.set("contype", "1")
    geom.set("conaffinity", "1")
    geom.set("friction", "0.8 0.02 0.01")
    geom.set("margin", "0.001")


def write_robot_xml(raw_xml: Path, out_xml: Path) -> None:
    import mujoco

    tree = ET.parse(raw_xml)
    root = tree.getroot()
    _ensure_option_defaults(root)
    _disable_mesh_collision(root)
    _add_collision_geoms(root)
    _add_sites_and_camera(root)
    _set_joint_dynamics(root)
    _indent(root)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="unicode", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(out_xml))
    tree = ET.parse(out_xml)
    root = tree.getroot()
    _add_actuators_sensors_keyframe(root, model)
    _indent(root)
    tree.write(out_xml, encoding="unicode", xml_declaration=True)


def _retarget_home_keyframe(robot_path: Path, scene_path: Path) -> float:
    """Lower/raise the freejoint so foot spheres rest on the floor."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    bottoms = []
    for name in FOOT_GEOM_NAMES:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        bottoms.append(float(data.geom_xpos[gid][2] - model.geom_size[gid][0]))
    if not bottoms:
        return HOME_HEIGHT
    dz = 0.002 - min(bottoms)
    tree = ET.parse(robot_path)
    key = tree.getroot().find("./keyframe/key")
    if key is None:
        return HOME_HEIGHT
    qpos = [float(v) for v in key.get("qpos", "").split()]
    qpos[2] += dz
    key.set("qpos", " ".join(f"{v:.6f}" for v in qpos))
    tree.write(robot_path, encoding="unicode", xml_declaration=True)
    return qpos[2]


def write_scene_xml(robot_file: Path, out_xml: Path) -> None:
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="quadruped_scene">
  <include file="{robot_file.name}"/>

  <statistic center="0 0 0.4" extent="1.4"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.0 0.02 0.01"/>
  </worldbody>
</mujoco>
"""
    out_xml.write_text(xml)


def convert(src_urdf: Path | None = None) -> Path:
    src = Path(src_urdf) if src_urdf is not None else original_urdf()
    if not src.is_file():
        src = cleaned_urdf()
    if not src.is_file():
        raise FileNotFoundError(f"No URDF found at {original_urdf()} or {cleaned_urdf()}")
    if not mesh_dir().is_dir():
        raise FileNotFoundError(f"Mesh directory missing: {mesh_dir()}")

    if src.resolve() == cleaned_urdf().resolve():
        urdf = src
    else:
        urdf = write_cleaned_urdf(src, cleaned_urdf())
    raw = mjcf_dir() / "_compiled_raw.xml"
    _compile_urdf(urdf, raw)
    write_robot_xml(raw, robot_xml())
    write_scene_xml(robot_xml(), scene_xml())
    z = _retarget_home_keyframe(robot_xml(), scene_xml())
    print(f"home_height={z:.3f} m")
    raw.unlink(missing_ok=True)
    return scene_xml()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=None, help="Override source URDF")
    args = parser.parse_args(argv)
    path = convert(args.urdf)
    print(f"Wrote {path}")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
