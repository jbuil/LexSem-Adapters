# Sección 4.1: Análisis comparativo de F-score promedio con y sin adapters

# EVALution + RoBERTa large
f_scores_with_adapters <- c(0.779679118379529, 0.784661322860552, 0.7864634681, 0.7892218638, 0.7863122961)
f_scores_without_adapters <- c(0.7784685738, 0.7669988804, 0.7757258741, 0.7734787974, 0.7640474697)

welch_test_result <- t.test(f_scores_with_adapters, f_scores_without_adapters, var.equal = FALSE)

print(welch_test_result)

# ROOT09 + RoBERTa large
f_scores_with_adapters <- c(0.9447611057, 0.938632812872537, 0.940226100411158, 0.937562827095718, 0.938250757772209)
f_scores_without_adapters <- c(0.934396487124733, 0.935323775499364, 0.931016215394144, 0.93727237878016, 0.93557331717452)

welch_test_result <- t.test(f_scores_with_adapters, f_scores_without_adapters, var.equal = FALSE)

print(welch_test_result)

# CogALex-V + RoBERTa large
f_scores_with_adapters <- c(0.7161987325, 0.7093759596, 0.7220533134, 0.7453981244, 0.6686879473)
f_scores_without_adapters <- c(0.7527281409, 0.7207655949, 0.7553925698, 0.756606709, 0.7177008248)

welch_test_result <- t.test(f_scores_with_adapters, f_scores_without_adapters, var.equal = FALSE)

print(welch_test_result)

# EVALution + RoBERTa base
f_scores_with_adapters <- c(0.7220569655, 0.7259030826, 0.7247173373, 0.7199218871, 0.7119481842)
f_scores_without_adapters <- c(0.7541604761, 0.7390849703, 0.7513364761, 0.7372280035, 0.7625970603)

welch_test_result <- t.test(f_scores_with_adapters, f_scores_without_adapters, var.equal = FALSE)

print(welch_test_result)

# ROOT09 + RoBERTa base
f_scores_with_adapters <- c(0.9134146276, 0.9198532764, 0.9213377589, 0.9227191115, 0.9252158332)
f_scores_without_adapters <- c(0.9247745788, 0.9313641614, 0.9358822267, 0.9328159853, 0.9182365943)

welch_test_result <- t.test(f_scores_with_adapters, f_scores_without_adapters, var.equal = FALSE)

print(welch_test_result)

# CogALex-V + RoBERTa base
f_scores_with_adapters <- c(0.4726711712, 0.5265698018, 0.5027345316, 0.4495609177, 0.4949370962)
f_scores_without_adapters <- c(0.7096783724, 0.6768677514, 0.7107432712, 0.6879635194, 0.6616647507)

welch_test_result <- t.test(f_scores_with_adapters, f_scores_without_adapters, var.equal = FALSE)

print(welch_test_result)

# Sección 4.2.1: Análisis comparativo de F-score promedio por tipo de relación con y sin adapters

# EVALution Synonym
porcentajes_aciertos_with_adapters <- c(66.43, 64.62, 63.90, 62.09, 68.23)
porcentajes_aciertos_without_adapters <- c(64.98, 57.04, 57.04, 63.90, 56.32)

welch_test_result <- t.test(porcentajes_aciertos_with_adapters, porcentajes_aciertos_without_adapters, var.equal = FALSE)

print(welch_test_result)

# EVALution PartOf
porcentajes_aciertos_with_adapters <- c(77.24, 83.45, 75.86, 80.00, 80.00)
porcentajes_aciertos_without_adapters <- c(77.93, 81.38, 77.24, 75.17, 82.07)

welch_test_result <- t.test(porcentajes_aciertos_with_adapters, porcentajes_aciertos_without_adapters, var.equal = FALSE)

print(welch_test_result)

# EVALution MadeOf
porcentajes_aciertos_with_adapters <- c(68.60, 74.42, 67.44, 70.93, 63.95)
porcentajes_aciertos_without_adapters <- c(69.77, 75.58, 67.44, 72.09, 65.12)

welch_test_result <- t.test(porcentajes_aciertos_with_adapters, porcentajes_aciertos_without_adapters, var.equal = FALSE)

print(welch_test_result)

# EVALution IsA
porcentajes_aciertos_with_adapters <- c(69.72, 68.41, 72.55, 72.77, 69.28)
porcentajes_aciertos_without_adapters <- c(71.46, 69.28, 73.42, 71.46, 68.85)

welch_test_result <- t.test(porcentajes_aciertos_with_adapters, porcentajes_aciertos_without_adapters, var.equal = FALSE)

print(welch_test_result)

# EVALution HasProperty
porcentajes_aciertos_with_adapters <- c(87.89, 89.44, 89.75, 90.68, 90.68)
porcentajes_aciertos_without_adapters <- c(89.75, 90.37, 90.99, 88.51, 90.06)

welch_test_result <- t.test(porcentajes_aciertos_with_adapters, porcentajes_aciertos_without_adapters, var.equal = FALSE)

print(welch_test_result)

# EVALution HasA
porcentajes_aciertos_with_adapters <- c(83.10, 83.10, 80.99, 83.10, 83.80)
porcentajes_aciertos_without_adapters <- c(76.76, 79.58, 80.99,  73.94, 83.10)

welch_test_result <- t.test(porcentajes_aciertos_with_adapters, porcentajes_aciertos_without_adapters, var.equal = FALSE)

print(welch_test_result)

# EVALution Antonym
porcentajes_aciertos_with_adapters <- c(86.99, 87.23, 88.92, 87.23, 86.51)
porcentajes_aciertos_without_adapters <- c(85.54, 85.54, 87.23, 86.51, 86.02)

welch_test_result <- t.test(porcentajes_aciertos_with_adapters, porcentajes_aciertos_without_adapters, var.equal = FALSE)

print(welch_test_result)
