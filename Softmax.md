# The Softmax Function and it's Derivative

>We define n as the softmax output vector index and k as the softmax input vector index.

The softmax function is defined as:

\begin{equation}
f(x)[n] = y[n] = \frac{e^{x[n]}}{\sum_{}^{}e^{x[k]}}
\end{equation}

The derivative of softmax(x)[n] with respect to x[k] has to be divided into two cases:

>The case in which n equals k:

$$f'(x)[n] = y'[n] = \frac{ e^{x[n]} * { \sum_{}^{} e^{x[k]} } - e^{x[n]} * e^{x[k]}}{ (\sum_{}^{} e^{x[k]})^2 }$$

$$= \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } - \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } * \frac{ e^{x[k]} }{ \sum_{}^{} e^{x[k]} }$$

$$= \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } * 1 - \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } * \frac{ e^{x[k]} }{ \sum_{}^{} e^{x[k]} }$$

$$= \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } * (1 - \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} })$$

$$= y[n] * (1 - y[k])$$

>The case in which n and k are not equal:

$$f'(x)[n] = y'[n] = \frac{ 0 * { \sum_{}^{} e^{x[k]} } - e^{x[n]} * e^{x[k]}}{ (\sum_{}^{} e^{x[k]})^2 }$$

$$= 0 - \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } * \frac{ e^{x[k]} }{ \sum_{}^{} e^{x[k]} }$$

$$= \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } * 0  - \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } * \frac{ e^{x[k]} }{ \sum_{}^{} e^{x[k]} }$$

$$= \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } * (0 - \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} })$$

$$= y[n] * (0 - y[k])$$
