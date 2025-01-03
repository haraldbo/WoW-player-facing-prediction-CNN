# Predicting player facing from the World of Warcraft minimap player arrow

<img src="./animation.gif" alt="animation of minimap compass needle with predicted angle on top" width="300"/>

## About
- A CNN that predicts the player facing direction from the World of Warcraft minimap cursor.
- Network predicts $(\hat{x}, \hat{y})$ on the unit circle. Predicted player facing is obtained by $atan2(\hat{y}, \hat{x})$.
- Dataset consist of images of size 32x32 that have been annotated with the actual player facing, $\theta$. 
    - $x = cos(\theta)$ 
    - $y = sin(\theta)$
- Loss = $\sqrt{(\hat{x}-x)^2 + (\hat{y} - y)^2}$

![test](dataset10x10.png)
