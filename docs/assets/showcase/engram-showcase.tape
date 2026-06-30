# Optional real terminal capture (requires brew install vhs + GUI/display).
# Preferred headless path: cd server && uv run python ../scripts/record_showcase.py

Output showcase-demo.mp4

Set Shell "zsh"
Set FontSize 15
Set Width 1080
Set Height 640
Set Theme "Catppuccin Mocha"
Set Padding 24
Set TypingSpeed 35ms

Hide
Type "cd server && uv run engram showcase run"
Enter
Sleep 7s