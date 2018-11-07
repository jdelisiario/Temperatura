#limpar workspace
rm(list=ls())

#limpar tela
cat('\014')

#bibliotecas
library("RSNNS")

#---------------------------------------
#FUNCOES

#funcao de ativacao utilizada
sech2<-function(u) 
{
  ((2/(exp(u)+exp(-u)))*(2/(exp(u)+exp(-u))))
}

#funcao que ajusta outliers de qualquer amostra
ajustaOutliers <- function(x, na.rm = TRUE, ...) 
{
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  
  for(i in 1:length(y)) 
  {
    #caso o primeiro valor seja NA procura o proximo valor nao NA e coloca
    #no lugar do NA
    if (is.na(y[1]) == TRUE)
    {
      encontrou = FALSE
      cont = 1
      posterior = NA
      #procura o primeiro numero POSTERIOR ao valor atual que nao seja NA
      while (encontrou == FALSE)
      {
        if (is.na(y[1+cont]) == TRUE)
        {
          cont <- cont + 1
        }
        else
        {
          posterior <- y[1+cont];
          encontrou <- TRUE
        }
      }
      
      y[1] <- posterior
    }
    
    #caso o ultimo valor seja NA procura o primeiro valor anterior que nao NA e coloca
    #no lugar do NA
    if (is.na(y[length(y)]) == TRUE)
    {
      encontrou <- FALSE
      cont <- 1
      anterior <- NA
      
      #procura o primeiro numero ANTERIOR ao valor atual que nao seja NA
      while (encontrou == FALSE)
      {
        if (is.na(y[length(y)-cont]) == TRUE)
        {
          cont <- cont + 1
        }
        else
        {
          anterior <- y[length(y)-cont];
          encontrou <- TRUE
        }
      }
      
      y[length(y)] <- anterior
    }
    
    
    
    if (is.na(y[i])==TRUE)
    {
      encontrou <- FALSE
      cont <- 1
      anterior <- NA
      
      #procura o primeiro numero ANTERIOR ao valor atual que nao seja NA
      while (encontrou == FALSE)
      {
        if (is.na(y[i-cont]) == TRUE)
        {
          cont <- cont + 1
        }
        else
        {
          anterior <- y[i-cont];
          encontrou <- TRUE
        }
      }
      
      encontrou = FALSE
      cont = 1
      posterior = NA
      
      #procura o primeiro numero POSTERIOR ao valor atual que nao seja NA
      while (encontrou == FALSE)
      {
        if (is.na(y[i+cont]) == TRUE)
        {
          cont <- cont + 1
        }
        else
        {
          posterior <- y[i+cont];
          encontrou <- TRUE
        }
      }
      
      #executa uma media entre o anterior e posterior valor valido na serie e insere no lugar do outlier
      y[i] <- (anterior+posterior)/2
    }
  }
  
  return(y)
}

#Coloca amostra nos intervalos proporcionais entre 0 e 1
padroniza <- function(s)
{
  retorno <- (s - min(s))/(max(s)-min(s))
  return(retorno)
}

carrega_subset <- function(arquivo,inicio,fim) 
{
  dados <- read.table(arquivo,
                      header=TRUE,
                      sep=";",
                      colClasses=c("character", "character", "character", rep("numeric",2)),
                      na="?")
  
  # converte para datetime
  #dados$Hora <- strptime(paste(dados$Data, dados$Hora), "%d/%m/%Y %H:%M:%S")
  dados$Data <- as.Date(dados$Data, "%d/%m/%Y")
  
  # separa o subset que vai de <inicio> a <fim>
  #dates <- as.Date(c(inicio, fim), "%Y-%m-%d")
  #dados <- subset(dados, Data %in% dates)
  
  return(dados)
}

despadroniza <- function(x)
{
  x*(37.1 - 22) + 22
  
}
#----------------------------------------------------------------
#Carregamento do dataset

temperaturas <- carrega_subset("temperaturas.csv", "2011-01-01","2015-01-01")
#separação da temperatura e da umidade
temp <- temperaturas$TempMaxima
umi <- temperaturas$Umidade.Relativa.Media

plot(temp, type="l")
plot(umi, type="l")

#frequência
hist(temp)
hist(umi)

#boxplot
boxplot(temp)
boxplot(umi)

#ajusta os outliers das 2 séries 
temp <- ajustaOutliers(temp)
umi <- ajustaOutliers(umi)

plot(temp, type="l")
plot(umi, type="l")

#frequência
hist(temp)
hist(umi)

#boxplot
boxplot(temp)
boxplot(umi)

#Padronização das 2 séries
temp <- padroniza(temp)
umi <- padroniza(umi)

plot(temp, type="l")
plot(umi, type="l")

#frequência
hist(temp)
hist(umi)

#boxplot
boxplot(temp)
boxplot(umi)

#separação dos dados do ano de 2015
tempeumi_2015 <- temperaturas[1458:1822,]
temp15ajus <- tempeumi_2015$TempMaxima 
umi15ajus <- tempeumi_2015$Umidade.Relativa.Media

temp15ajus <- ajustaOutliers(temp15ajus)
umi15ajus <- ajustaOutliers(umi15ajus)

temp15ajus <- padroniza(temp15ajus)
umi15ajus <- padroniza(umi15ajus)

#mostra a curva do ano de 2015
plot(temp15ajus, type="l")
plot(umi15ajus, type="l")

#frequência
hist(temp15ajus)
hist(umi15ajus)

#boxplot
boxplot(temp15ajus)
boxplot(umi15ajus)

#data15 <- tempeumi_2015$Data[1:365]

#ANALISE DE DADOS
#analise da serie de umidade em relação a data
plot(tempeumi_2015$Data, temp15ajus,
     type="l",
     xlab="data",
     ylab="temperatura")

#analise da serie de umidade em relação a data
plot(tempeumi_2015$Data, umi15ajus,
     type="l",
     xlab="data",
     ylab="umidade")

#IMPLEMENTACAO DA REDE NEURAL NOS DADOS AJUSTADOS
x1<-padroniza(as.numeric(tempeumi_2015$Data))
x2<-padroniza(umi15ajus)

x <- cbind(x1,x2)

y<-padroniza(temp15ajus)

#configuracoes da MLP
nNeuronios = 5
maxEpocas <- 15000

#treinamento da MLP
redeCA <- NULL

#Backpropagation
print("treinando a rede na serie ajustada...")

redeCA<-mlp(x, y, size=nNeuronios, maxit=maxEpocas, initFunc="Randomize_Weights",
            initFuncParams=c(-0.3, 0.3), learnFunc="Std_Backpropagation",
            learnFuncParams=c(0.05), updateFunc="Topological_Order",
            updateFuncParams=c(0), hiddenActFunc="Act_Logistic",
            shufflePatterns=F, linOut=TRUE)

plot(redeCA$IterativeFitError,type="l",main="Erro da MLP CA")

#EXECUTANDO A PREVISOES COM O MODELO TREINADO
#Carregando as informações de 2016
tempeumi2016 <- temperaturas[1822:2171,]
#umidade de 2016
umi16 <- tempeumi2016$Umidade.Relativa.Media
umi16ajus <- ajustaOutliers(umi16)

temp16ajus <- tempeumi2016$TempMaxima

z1<-padroniza(as.numeric(tempeumi2016$Data))
z2<-padroniza(umi16ajus)

z <- cbind(z1,z2)

y<-temp16ajus

yhat = vector()
for (i in 1:length(z[,1]))
{
  print(i)
  yhat[i] = predict(redeCA,z[i,])
}

yhat <- despadroniza(yhat)

#CALCULO DO ERRO
erromlpca <- mean(sqrt((y-yhat)^2))

print(paste("ERRO MLP CA:",erromlpca))

#previsao
plot(tempeumi2016$Data, yhat, col="blue", type="l", ylab="", xaxt='n', yaxt='n')
par(new=T)
#esperado dos dados reais
plot(tempeumi2016$Data, y, col="red", type="l", ylab="", xaxt='n', yaxt='n')
