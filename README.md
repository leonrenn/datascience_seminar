# datascience_seminar
Results and insights collected from the data science seminar by Lukas Heinrich.

## ParameterizedNeuralNetwork
In ***High Energy Physics*** it is common that one has to decide if the data that is taken is from some specific signal ( $H_{a}$ ) or from the background ( $H_{0}$ ).<br>
This is simple binary hypothesis testing. For big experiments and complicated processes the likelihood may not be known and therefore has to be learned. <br>
Neural Network provide a very nice approach to this problem. In this case, a NN is trained on parameterized (means as hypothesis) on 1 and 2 dimesional data.

## NormalizingFlows
In physics, it is often the case that the exact probability distribution to a problem might not be known (e.g., after complicated processes of particle scatterings). The pdf and the underlying transformation can both be learned by a NN. Because of keeping the density equal to unity this flow is called normalizing. This is shown via 1 dimensional data.
