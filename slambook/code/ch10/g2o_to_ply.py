#!/usr/bin/env python3
import argparse
from pathlib import Path


def read_vertices(g2o_path):
    vertices = []
    with open(g2o_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "VERTEX_SE3:QUAT":
                continue
            if len(parts) < 9:
                continue
            idx = int(parts[1])
            x, y, z = map(float, parts[2:5])
            vertices.append((idx, x, y, z))
    vertices.sort(key=lambda item: item[0])
    return vertices


def write_ply(vertices, ply_path, color):
    r, g, b = color
    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for _, x, y, z in vertices:
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def parse_color(value):
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be R,G,B")
    color = tuple(int(p) for p in parts)
    if any(c < 0 or c > 255 for c in color):
        raise argparse.ArgumentTypeError("color values must be in [0, 255]")
    return color


def main():
    parser = argparse.ArgumentParser(description="Convert SE3 g2o vertices to a PLY point cloud.")
    parser.add_argument("input", help="input .g2o file")
    parser.add_argument("output", nargs="?", help="output .ply file")
    parser.add_argument("--color", type=parse_color, default=(0, 255, 0), help="vertex color as R,G,B")
    args = parser.parse_args()

    g2o_path = Path(args.input)
    ply_path = Path(args.output) if args.output else g2o_path.with_suffix(".ply")

    vertices = read_vertices(g2o_path)
    if not vertices:
        raise SystemExit(f"No VERTEX_SE3:QUAT entries found in {g2o_path}")

    write_ply(vertices, ply_path, args.color)
    print(f"Wrote {len(vertices)} vertices to {ply_path}")


if __name__ == "__main__":
    main()
