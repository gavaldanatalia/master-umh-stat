import Data.List (nub)

-- Calcula la matriz de confusión para un modelo de reglas
testModel :: [Rule] -> Dataset -> Int -> ConfMatrix
testModel rules testData classPosition =
    foldl updateConfMatrix [] testData
  where
    -- Actualiza la matriz de confusión para cada fila
    updateConfMatrix :: ConfMatrix -> Row -> ConfMatrix
    updateConfMatrix confMatrix row =
        let realClass = row !! classPosition
            predictedClass = predict rules row classPosition
        in addToMatrix confMatrix realClass predictedClass

    -- Agrega una entrada a la matriz de confusión
    addToMatrix :: ConfMatrix -> String -> String -> ConfMatrix
    addToMatrix confMatrix real pred =
        case lookup (real, pred) (map (\(r, p, c) -> ((r, p), c)) confMatrix) of
            Nothing -> (real, pred, 1) : confMatrix
            Just count -> map (\(r, p, c) ->
                if r == real && p == pred
                then (r, p, c + 1)
                else (r, p, c)) confMatrix
