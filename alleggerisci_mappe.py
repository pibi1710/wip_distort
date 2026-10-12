# WIP - Distorci web: converte le mappe di profondità a 16 bit in PNG a 8 bit.
# Il browser legge comunque solo 8 bit: stessa resa, file circa 20 volte più leggeri.
# Le mappe già a 8 bit vengono lasciate stare. Lo lancia avvia.command.
from pathlib import Path
import numpy as np
from PIL import Image

for p in sorted((Path(__file__).parent / 'Materiali' / 'Maps_distorci').glob('*.png')):
	im = Image.open(p)
	if im.mode == 'L':
		continue
	a = np.array(im).astype(np.float64)
	if a.max() > 255:
		a = a / 257
	Image.fromarray(a.round().clip(0, 255).astype(np.uint8), 'L').save(p, optimize=True)
	print('mappa alleggerita:', p.name)
