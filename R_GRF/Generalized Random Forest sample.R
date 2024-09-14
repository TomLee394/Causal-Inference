library(dplyr)

library(pROC)

library(tidyr)

library(grf)

library(ggplot2)

library(reshape2)



#load data

df_final <- read.csv("data.csv")



#split data into test and train

smp_size <- floor(0.95 * nrow(df_final))



## set the seed to make your partition reproducible

set.seed(123)

train_ind <- sample(seq_len(nrow(df_final)), size = smp_size)



df_train <- df_final[train_ind, ]

df_test <- df_final[-train_ind, ]



#Select the variables with a high degree of correlation as observed in the correlation matrix

X.temp <- df_final %>%
  
  select(age, yr_income, revolve_pct, debit_active_pct,
         
         avg_rtl_rvlv, ref_fico, mob, cr_lmt, insurance_flag, unique_6mo_MCC
         
         #             assist
         
  )



X.train <- X.temp[train_ind, ]

X.test <- X.temp[-train_ind, ]

Y.test <- df_test %>% select(rtl_3mo_rvlv_target)

Y <- df_train %>% select(rtl_3mo_rvlv_target)

W <- df_train %>% select(IP_acct_flag)



X <- data.matrix(X.train)

Y <- data.matrix(Y)

W <- data.matrix(W)



aggregate(x = df_final$rtl_3mo_rvlv_target,                # Specify data column
          
          by = list(df_final$IP_acct_flag),              # Specify group indicator
          
          FUN = mean) 



#Fit models for Y and W

Y.forest <- regression_forest(X, Y, tune.parameters = "all")

Y.hat <- predict(Y.forest)$predictions

W.forest <- regression_forest(X, W, tune.parameters = "all")

W.hat <- predict(W.forest)$predictions



# Variable selection, if needed:

# cf.raw <- causal_forest(X, Y, W, Y.hat=Y.hat, W.hat=W.hat, num_trees = 3000)

# varimp <- variable_importance(cf.raw)

# selected.idx <- which(varimp > mean(varimp))

# varimp <- data.frame(varimp=variable_importance(cf.raw), X=colnames(X))

# varimp$varimp <- as.numeric(as.character(varimp$varimp))



#train a causal forest

tau.forest <- causal_forest(X, Y, W, Y.hat=Y.hat, W.hat=W.hat, num.trees = 4000)

tau.hat <- predict(tau.forest, X.test, estimate.variance = TRUE)

sigma.hat <- sqrt(tau.hat$variance.estimates)



#Estimate the conditional average treatment effect (CATE) over all observations

ate.all <- average_treatment_effect(tau.forest, target.sample = "treated")

paste("95% CI for the ATE (all):", round(ate.all[1], 2), "+/-", round(qnorm(0.975) * ate.all[2], 2))

paste("95% CI for the ATE (all):", round(ate.all[1] - qnorm(0.975) * ate.all[2], 2), ",", round(ate.all[1] + qnorm(0.975) * ate.all[2], 2))

paste("P-value", round(pnorm(ate.all[1], mean = 0, sd = ate.all[2], lower.tail = FALSE, log.p = FALSE)*2, 4))