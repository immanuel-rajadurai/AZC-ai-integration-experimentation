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
        const modelJson = require('./assets/model/model.json');

        const shard1 = require('./assets/model/group1-shard1of25.bin');
        const shard2 = require('./assets/model/group1-shard2of25.bin');
        const shard3 = require('./assets/model/group1-shard3of25.bin');
        const shard4 = require('./assets/model/group1-shard4of25.bin');
        const shard5 = require('./assets/model/group1-shard5of25.bin');
        const shard6 = require('./assets/model/group1-shard6of25.bin');
        const shard7 = require('./assets/model/group1-shard7of25.bin');
        const shard8 = require('./assets/model/group1-shard8of25.bin');
        const shard9 = require('./assets/model/group1-shard9of25.bin');
        const shard10 = require('./assets/model/group1-shard10of25.bin');
        const shard11 = require('./assets/model/group1-shard11of25.bin');
        const shard12 = require('./assets/model/group1-shard12of25.bin');
        const shard13 = require('./assets/model/group1-shard13of25.bin');
        const shard14 = require('./assets/model/group1-shard14of25.bin');
        const shard15 = require('./assets/model/group1-shard15of25.bin');
        const shard16 = require('./assets/model/group1-shard16of25.bin');
        const shard17 = require('./assets/model/group1-shard17of25.bin');
        const shard18 = require('./assets/model/group1-shard18of25.bin');
        const shard19 = require('./assets/model/group1-shard19of25.bin');
        const shard20 = require('./assets/model/group1-shard20of25.bin');
        const shard21 = require('./assets/model/group1-shard21of25.bin');
        const shard22 = require('./assets/model/group1-shard22of25.bin');
        const shard23 = require('./assets/model/group1-shard23of25.bin');
        const shard24 = require('./assets/model/group1-shard24of25.bin');
        const shard25 = require('./assets/model/group1-shard25of25.bin');

        

        const combinedWeights = [
          shard1, shard2, shard3, shard4, shard5,
          shard6, shard7, shard8, shard9, shard10,
          shard11, shard12, shard13, shard14, shard15,
          shard16, shard17, shard18, shard19, shard20,
          shard21, shard22, shard23, shard24, shard25
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

