(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("microtype" "final" "tracking=smallcaps" "expansion=alltext" "protrusion=true") ("siunitx" "separate-uncertainty")))
   (add-to-list 'LaTeX-verbatim-environments-local "semiverbatim")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "inputenc"
    "fontenc"
    "graphicx"
    "booktabs"
    "xspace"
    "microtype"
    "amsmath"
    "amsfonts"
    "amssymb"
    "bm"
    "mathtools"
    "nicefrac"
    "siunitx"
    "calc")
   (TeX-add-symbols
    "lasso"
    "fista"
    "ista")
   (LaTeX-add-labels
    "eq:optGoal"
    "eq:diagonal-prior"
    "eq:tik-langrangian"
    "eq:diagonal-matrix"
    "fig:reg-pdfs"
    "fig:sparse-results"
    "fig:results-opt"))
 :latex)

