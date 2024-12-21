# Predicting player facing from the World of Warcraft minimap compass arrow using a CNN

<img src="./animation.gif" alt="animation of minimap compass needle with predicted angle on top" width="200"/>

## What it does
- A CNN that predicts the player facing direction from the World of Warcraft minimap compass arrow.
- Network outputs (x, y) on the unit circle, and facing direction is attained by $atan2(y, x)$.
