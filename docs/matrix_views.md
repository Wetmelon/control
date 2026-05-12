# Matrix Views

Parent matrix used in all examples:

$$A = \begin{bmatrix}1&2&3&4\\5&6&7&8\\9&10&11&12\\13&14&15&16\end{bmatrix}$$

---

$$
\begin{array}{ll}
\textbf{Diagonal}\quad d(i)=A_{i,i}
&
\textbf{UpperTriangle}\quad U(r,c)=A_{r,c},\; r \le c
\\[4pt]
\begin{bmatrix}\mathbf{1}&\cdot&\cdot&\cdot\\\cdot&\mathbf{6}&\cdot&\cdot\\\cdot&\cdot&\mathbf{11}&\cdot\\\cdot&\cdot&\cdot&\mathbf{16}\end{bmatrix}
&
\begin{bmatrix}\mathbf{1}&\mathbf{2}&\mathbf{3}&\mathbf{4}\\\cdot&\mathbf{6}&\mathbf{7}&\mathbf{8}\\\cdot&\cdot&\mathbf{11}&\mathbf{12}\\\cdot&\cdot&\cdot&\mathbf{16}\end{bmatrix}
\\[18pt]
\\
\textbf{LowerTriangle}\quad L(r,c)=A_{r,c},\; r \ge c
&

\textbf{RowView}\quad k=1,\; v(c)=A_{1,c}
\\[4pt]
\begin{bmatrix}\mathbf{1}&\cdot&\cdot&\cdot\\\mathbf{5}&\mathbf{6}&\cdot&\cdot\\\mathbf{9}&\mathbf{10}&\mathbf{11}&\cdot\\\mathbf{13}&\mathbf{14}&\mathbf{15}&\mathbf{16}\end{bmatrix}
&
\begin{bmatrix}\cdot&\cdot&\cdot&\cdot\\\mathbf{5}&\mathbf{6}&\mathbf{7}&\mathbf{8}\\\cdot&\cdot&\cdot&\cdot\\\cdot&\cdot&\cdot&\cdot\end{bmatrix}
\\[18pt]
\\
\textbf{ColView}\quad k=2,\; v(r)=A_{r,2}
&
\textbf{TransposeView}\quad T_{r,c}=A_{c,r}
\\[4pt]
\begin{bmatrix}\cdot&\cdot&\mathbf{3}&\cdot\\\cdot&\cdot&\mathbf{7}&\cdot\\\cdot&\cdot&\mathbf{11}&\cdot\\\cdot&\cdot&\mathbf{15}&\cdot\end{bmatrix}
&
\begin{bmatrix}1&5&9&13\\2&6&10&14\\3&7&11&15\\4&8&12&16\end{bmatrix}
\\[18pt]
\\
\textbf{Block}\quad r_0=1,\; c_0=1,\; 2{\times}2 \quad B(r,c)=A_{r_0+r,\;c_0+c}
\\[4pt]
\begin{bmatrix}\cdot&\cdot&\cdot&\cdot\\\cdot&\mathbf{6}&\mathbf{7}&\cdot\\\cdot&\mathbf{10}&\mathbf{11}&\cdot\\\cdot&\cdot&\cdot&\cdot\end{bmatrix}
\end{array}
$$
