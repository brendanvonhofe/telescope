(TeX-add-style-hook
 "pitch"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "15pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "inputenc"
    "fontenc"
    "graphicx"
    "grffile"
    "longtable"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "textcomp"
    "amssymb"
    "capt-of"
    "hyperref"
    "helvet")
   (LaTeX-add-labels
    "sec:orgb5dae43"
    "sec:org3168d0f"
    "sec:org18f8651"
    "sec:orgbc621ed"
    "sec:org0cd51f8"
    "sec:org9aa1bbb"
    "sec:orgab6cf12"
    "sec:org391378b"
    "sec:orge205cba"
    "sec:orgb6f315a"
    "sec:org5549106"
    "sec:orgbfa8ebf"
    "sec:orgf1da5b0"
    "sec:orga28108c"
    "sec:orgac17977"
    "sec:orgb665fea"
    "sec:org4a5e17a"
    "sec:orgfe46f40"
    "sec:orga07182d"
    "sec:orge8ca56b"
    "sec:orgc52e48f"
    "sec:orgd947613"
    "sec:org04e1b6e"
    "sec:orgc406f85"
    "sec:org45ebf18"
    "sec:org953dd5d"
    "sec:orgddd90d4"
    "sec:org47af881"
    "sec:org196c672"))
 :latex)

