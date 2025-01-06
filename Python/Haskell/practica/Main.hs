import Predict

main :: IO ()
main = do
    -- Carga el dataset desde el archivo CSV
    dataset <- loadCSV "datos.csv"

    -- Define las reglas adaptadas al dataset
    let rules = [
            ( (30, ["AB", "BB"]),
              [ (25, "A")
              , (5, "F")
              ]
            ),
            ( (20, ["AC", "CC"]),
              [ (18, "A")
              , (2, "F")
              ]
            ),
            ( (15, ["AA", "BD"]),
              [ (12, "A")
              , (3, "H")
              ]
            )
          ]

    -- Clasificación de cada fila del dataset
    putStrLn "Clasificación por fila del dataset:"
    mapM_ (\row -> do
        let predictedClass = predict rules row 10 -- Ajusta la posición según el CSV
        putStrLn $ "Fila: " ++ show row ++ " -> Clase predicha: " ++ predictedClass
      ) dataset

    -- Calcula la matriz de confusión
    let confMatrix = testModel rules dataset 10 -- Ajusta la posición según el CSV
    putStrLn "\nMatriz de Confusión:"
    mapM_ print confMatrix

    -- Realiza Cross-Validation
    let k = 5 -- Número de particiones
    let crossData = crossValidationData dataset k

    putStrLn "\nResultados de Cross-Validation:"
    let crossResults = map (\(train, test) -> testModel rules test 10) crossData
    mapM_ (\(i, conf) -> do
        putStrLn $ "\nIteración " ++ show i ++ ":"
        mapM_ print conf
      ) (zip [1..] crossResults)

    -- Suma de todas las matrices de confusión en Cross-Validation
    let totalConfMatrix = sumConfMatrix crossResults
    putStrLn "\nMatriz de Confusión Total (Cross-Validation):"
    mapM_ print totalConfMatrix