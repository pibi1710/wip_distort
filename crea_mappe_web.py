# WIP - Distorci web: prende tutte le foto del sito WIP, crea le mappe di
# profondità che mancano e aggiorna l'elenco usato dalla pagina.
#
# Le foto NON vengono copiate qui: la pagina le carica direttamente dal sito WIP
# (https://pibi1710.github.io/uploads/...), nella misura più grande pubblicata
# (al massimo 2048 px). L'elenco delle foto si legge dalla copia di deploy del
# sito (~/Sites/week-in-photography), che è identica a quella online.
# Le mappe (8 bit grigio, chiaro = vicino) stanno in Materiali/Maps_distorci/,
# con il nome del file originale di Instatic; quelle di foto tolte dal sito
# vengono eliminate. Stesso modello della patch TouchDesigner, ma i file della
# patch non vengono toccati. Lo lancia pubblica-foto.command.
#
# Lancio a mano (serve l'ambiente Python della patch, che ha torch e transformers):
#   ~/Desktop/WIP/Touchdesigner/Depth/.venv/bin/python crea_mappe_web.py

import html
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# cartella di Distorci web: quella di questo file, oppure quella passata come argomento (per le prove)
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
SITO = Path.home() / 'Sites/week-in-photography'
URL_SITO = 'https://pibi1710.github.io'
MAPPE = BASE / 'Materiali' / 'Maps_distorci'
ELENCO = BASE / 'Materiali' / 'elenco.json'
MODELLO = 'depth-anything/Depth-Anything-V2-Large-hf'  # licenza CC BY-NC 4.0: solo uso non commerciale
LARGHEZZA_MAX = 2048


def foto_del_sito():
	"""Per ogni foto del sito: nome (file originale senza estensione) e variante più grande pubblicata."""
	pagina = (SITO / 'index.html').read_text()
	foto = {}
	for tag in re.findall(r'<img[^>]*class="photo[^"]*"[^>]*>', pagina):
		src = re.search(r'\ssrc="([^"]+)"', tag)
		srcset = re.search(r'\ssrcset="([^"]+)"', tag)
		if not src or not srcset:
			continue
		nome = Path(html.unescape(src.group(1))).stem
		if nome in foto:
			continue
		varianti = []
		for voce in html.unescape(srcset.group(1)).split(','):
			percorso, larghezza = voce.split()
			varianti.append((int(larghezza.rstrip('w')), percorso))
		# la più grande entro il tetto, tra quelle davvero presenti nella copia di deploy
		for larghezza, percorso in sorted(varianti, reverse=True):
			if larghezza <= LARGHEZZA_MAX and (SITO / percorso.lstrip('/')).exists():
				foto[nome] = percorso
				break
		else:
			print('  saltata (nessuna variante pubblicata):', nome)
	return foto


def crea_mappe(da_fare):
	import torch
	from transformers import pipeline
	device = 'mps' if torch.backends.mps.is_available() else 'cpu'
	stima = pipeline('depth-estimation', model=MODELLO, device=device)
	for n, (nome, percorso) in enumerate(da_fare, 1):
		img = ImageOps.exif_transpose(Image.open(SITO / percorso.lstrip('/'))).convert('RGB')
		d = stima(img)['predicted_depth']
		# riporta la mappa alla risoluzione della foto pubblicata
		d = torch.nn.functional.interpolate(
			d.reshape(1, 1, *d.shape[-2:]), size=(img.height, img.width),
			mode='bicubic', align_corners=False)[0, 0].cpu().numpy()
		d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
		# file temporaneo e poi rinomina: mai una mappa scritta a metà
		finale = MAPPE / (nome + '.png')
		temp = MAPPE / ('.' + nome + '.tmp.png')
		Image.fromarray((d * 255).round().astype(np.uint8), 'L').save(temp, optimize=True)
		temp.replace(finale)
		print(f'  mappa creata ({n}/{len(da_fare)}):', finale.name)


def main():
	MAPPE.mkdir(parents=True, exist_ok=True)
	foto = foto_del_sito()
	print(f'Foto sul sito WIP: {len(foto)}')

	da_fare = [(nome, p) for nome, p in sorted(foto.items()) if not (MAPPE / (nome + '.png')).exists()]
	if da_fare:
		print(f'Creo {len(da_fare)} mappe (circa 1 secondo a foto, più il caricamento del modello)...')
		crea_mappe(da_fare)
	else:
		print('Nessuna foto nuova: tutte le mappe sono già pronte.')

	# mappe di foto tolte dal sito: non servono più alla pagina
	for m in sorted(MAPPE.glob('*.png')):
		if m.stem not in foto:
			m.unlink()
			print('  mappa tolta (la foto non è più sul sito):', m.name)

	immagini = [{'foto': URL_SITO + p, 'mappa': nome + '.png'}
	            for nome, p in sorted(foto.items()) if (MAPPE / (nome + '.png')).exists()]
	ELENCO.write_text(json.dumps({'immagini': immagini}, indent=1))
	print(f'Elenco aggiornato: {len(immagini)} foto con mappa.')


if __name__ == '__main__':
	main()
