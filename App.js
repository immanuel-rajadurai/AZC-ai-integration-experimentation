import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Platform, Image as RNImage } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as jpeg from 'jpeg-js';
import * as FileSystem from 'expo-file-system';
import { Asset } from 'expo-asset';
import labels from './assets/labels.json'; 
import peacock from './assets/peacock.jpg';


export default function App() {
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState(null);

  useEffect(() => {
    async function loadModel() {

      try {
        await tf.ready();
        const modelJson = require('./assets/mobilenet_model/model.json');

        const shard1 = require('./assets/mobilenet_model/group1-shard1of3.bin');
        const shard2 = require('./assets/mobilenet_model/group1-shard2of3.bin');
        const shard3 = require('./assets/mobilenet_model/group1-shard3of3.bin');

        

        const combinedWeights = [
          shard1, shard2, shard3, 
        ];
        console.log("Model loaded." )



        console.log("creating graph model")
        const loadedModel = await tf.loadGraphModel(bundleResourceIO(modelJson, combinedWeights));
        console.log("successfully created graph model")
        setModel(loadedModel);

      } catch (err) {
        console.error(err);
      }
    }

    loadModel();
  }, []);


   function decodeImage(imageData) {
    const pixels = jpeg.decode(imageData, true); 
    const tensor = tf.browser.fromPixels(pixels); 
    return tensor;
  }


  useEffect(() => {
    async function classifyImage() {
      if (model) {
        try {

          const asset = Asset.fromModule(peacock);
          await asset.downloadAsync();
          const imageUri = asset.localUri || asset.uri;

  
          const base64String = await FileSystem.readAsStringAsync(imageUri, {
            encoding: FileSystem.EncodingType.Base64,
          });


          const imageBuffer = Uint8Array.from(atob(base64String), c => c.charCodeAt(0));

          const imageTensor = tf.tidy(() => {
            const decodedImage = decodeImage(imageBuffer);
            return decodedImage.resizeNearestNeighbor([224, 224]).toFloat().expandDims();
          });

          const prediction = await model.predict(imageTensor).data();
          const highestPredictionIndex = prediction.indexOf(Math.max(...prediction));
          const predictedClassEntry = labels[highestPredictionIndex];
          const predictedClass = predictedClassEntry ? predictedClassEntry[1] : 'Unknown'; // class name
          
        
          console.log('Predicted Class:', predictedClass);

          setPredictions(predictedClass);
        } catch (error) {
          console.error('Error classifying image:', error);
        }
      }
    }


    classifyImage();
  }, [model]);
  

  return (
    <View style={styles.container}>
      <RNImage
        source={peacock}
        style={styles.image}
      />
      {predictions && (
        <Text style={styles.predictions}>Predicted Animal: {predictions}</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  image: {
    width: 300,
    height: 300,
    margin: 20,
  },
  predictions: {
    fontSize: 16,
    marginTop: 20,
  },
  
});

