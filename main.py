from models.dummy import generate_unconditionally
from models.dummy import generate_conditionally

from utils import plot_stroke

strokes = generate_unconditionally()
plot_stroke(strokes, 'unconditional.png')
strokes = generate_conditionally()
plot_stroke(strokes, 'conditional.png')
