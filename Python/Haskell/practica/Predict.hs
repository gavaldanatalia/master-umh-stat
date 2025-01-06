module Predict (
    testModel,
    predict,
    Rule,
    Dataset,
    loadCSV,
    crossValidationData,
    sumConfMatrix
) where

import Data.List (maximumBy)
import System.IO (readFile)

-- Tipos
type ConfMatrix = [(String, String, Int)]
type Antecedent = (Int, [String])
type Consequent = [(Int, String)]
type Rule = (Antecedent, Consequent)
type Row = [String]
type Dataset = [Row]

-- Divide una cadena en valores separados por un delimitador
split :: Char -> String -> [String]
split _ "" = []
split sep str = let (word, rest) = break (== sep) str
                in word : if null rest then [] else split sep (drop 1 rest)

-- Lee un archivo CSV y elimina caracteres innecesarios
loadCSV :: FilePath -> IO Dataset
loadCSV filePath = do
    content <- readFile filePath
    return $ map (map strip . split ',') (lines content)

-- Función para eliminar caracteres innecesarios como '\r' o espacios
strip :: String -> String
strip = reverse . dropWhile (== '\r') . reverse

-- Predice la clase para una fila del dataset
predict :: [Rule] -> Row -> Int -> String
predict rules row classPosition =
    let matchingRules = filter (\rule -> matchRule rule row classPosition) rules
    in if null matchingRules
       then "Sin coincidencias"
       else snd $ maximumBy (\(conf1, _) (conf2, _) -> compare conf1 conf2) 
                            (concatMap snd matchingRules)

-- Comprueba si el antecedente de una regla coincide con una fila
matchRule :: Rule -> Row -> Int -> Bool
matchRule ((_, antecedent), _) row _ =
    all (`elem` row) antecedent

-- Calcula la matriz de confusión para un modelo de reglas
testModel :: [Rule] -> Dataset -> Int -> ConfMatrix
testModel rules testData classPosition =
    foldl updateConfMatrix [] testData
  where
    updateConfMatrix :: ConfMatrix -> Row -> ConfMatrix
    updateConfMatrix confMatrix row =
        let realClass = row !! classPosition
            predictedClass = predict rules row classPosition
        in addToMatrix confMatrix realClass predictedClass

    addToMatrix :: ConfMatrix -> String -> String -> ConfMatrix
    addToMatrix confMatrix real pred =
        case lookup (real, pred) (map (\(r, p, c) -> ((r, p), c)) confMatrix) of
            Nothing -> (real, pred, 1) : confMatrix
            Just count -> map (\(r, p, c) ->
                if r == real && p == pred
                then (r, p, c + 1)
                else (r, p, c)) confMatrix

-- Genera los conjuntos de entrenamiento y prueba para cross-validation
crossValidationData :: Dataset -> Int -> [(Dataset, Dataset)]
crossValidationData dataset k =
    let chunkSize = length dataset `div` k
        chunks = splitChunks chunkSize dataset
    in [ (concat (exclude i chunks), chunks !! i) | i <- [0..k-1] ]
  where
    -- Divide el dataset en k partes
    splitChunks :: Int -> Dataset -> [Dataset]
    splitChunks _ [] = []
    splitChunks size xs = take size xs : splitChunks size (drop size xs)

    -- Excluye el subconjunto en la posición i
    exclude :: Int -> [a] -> [a]
    exclude i xs = take i xs ++ drop (i + 1) xs

--Realizar el sumatorio de las matrices de confusión del proceso cross-validation
sumConfMatrix :: [ConfMatrix] -> ConfMatrix
sumConfMatrix matrices = foldl mergeMatrices [] matrices
  where
    mergeMatrices :: ConfMatrix -> ConfMatrix -> ConfMatrix
    mergeMatrices acc matrix =
        foldl (\acc (real, pred, count) ->
            case lookup (real, pred) (map (\(r, p, c) -> ((r, p), c)) acc) of
                Nothing -> (real, pred, count) : acc
                Just existingCount -> map (\(r, p, c) ->
                    if r == real && p == pred
                    then (r, p, c + count)
                    else (r, p, c)) acc
        ) acc matrix