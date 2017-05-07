FILE := main
OUT  := build


.PHONY: main.pdf latexmk clean purge

all: main.pdf

main.pdf: main.tex
	latexmk -outdir=$(OUT) -pdflatex="pdflatex -interaction=nonstopmode" -pdf $(FILE) #-use-make

clean:
	latexmk -CA

pdf:
	# Also see .latexmkrc
	latexmk -outdir=$(OUT) -pdf $(FILE)

#clean:
#	rm -rf $(filter-out $(OUT)/$(FILE).pdf, $(wildcard $(OUT)/*))

purge:
	rm -rf $(OUT)

