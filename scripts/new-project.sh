#!/usr/bin/env bash
#
# Scaffold a new numbered project for the ML/AI journey monorepo.
#
# Usage:
#   scripts/new-project.sh <slug> ["Human Title"] ["Concepts / techniques"]
#
# Example:
#   scripts/new-project.sh linear-regression-scratch \
#       "Linear Regression from Scratch" "gradient descent, no sklearn"
#
# It figures out the next NN- prefix, creates NN-<slug>/ with Data/ and Output/,
# a README from the house template, a uv-ready pyproject.toml, a starter main.py,
# and appends a row to the root README.md projects table.

set -euo pipefail
shopt -s nullglob

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

slug="${1:-}"
if [[ -z "$slug" ]]; then
  echo "Usage: $0 <slug> [\"Human Title\"] [\"Concepts\"]" >&2
  echo "  e.g. $0 cnn-mnist \"CNN on MNIST\" \"convolutions, PyTorch\"" >&2
  exit 1
fi

# --- next zero-padded number, scanning existing NN-* folders ---------------
last=0
for d in [0-9][0-9]-*/; do
  n=$((10#${d%%-*}))
  (( n > last )) && last=$n
done
next=$(printf "%02d" $((last + 1)))
dir="${next}-${slug}"

[[ -e "$dir" ]] && { echo "Error: $dir already exists" >&2; exit 1; }

title="${2:-$slug}"
concepts="${3:-TODO}"

mkdir -p "$dir/Data" "$dir/Output"
# Keep the (otherwise empty) dirs in git until real files land in them.
touch "$dir/Data/.gitkeep" "$dir/Output/.gitkeep"

cat > "$dir/pyproject.toml" <<EOF
[project]
name = "${dir}"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = []
EOF

cat > "$dir/main.py" <<EOF
"""${title}

TODO: describe the problem and approach.
"""


def main():
    raise NotImplementedError("TODO: build ${title}")


if __name__ == "__main__":
    main()
EOF

cat > "$dir/README.md" <<EOF
# ${next} · ${title}

> One-paragraph description of the problem and the approach. TODO.

## 🎯 Problem

TODO — what are we predicting / building?

## 🧠 Techniques & concepts

- ${concepts}

## ▶️ Running it

\`\`\`bash
uv sync
uv run python main.py
\`\`\`

## 💡 Lessons / ideas to revisit

- TODO
EOF

# --- append a row to the root README projects table ------------------------
# Inserts right after the last existing data row (a line beginning "| NN ").
row="| ${next} | [${title}](./${dir}) | ${concepts} | 🚧 |"
awk -v row="$row" '
  { lines[NR] = $0; if ($0 ~ /^\| [0-9]/) last = NR }
  END { for (i = 1; i <= NR; i++) { print lines[i]; if (i == last) print row } }
' README.md > README.md.tmp && mv README.md.tmp README.md

echo "Created $dir and added it to the README table."
echo
echo "Next steps:"
echo "  cd $dir"
echo "  uv add <packages>      # e.g. uv add numpy scikit-learn"
echo "  \$EDITOR README.md main.py"
