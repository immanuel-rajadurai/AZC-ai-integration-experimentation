import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Image } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';
import { fetch } from '@tensorflow/tfjs-react-native';

export default function App() {
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState(null);

  useEffect(() => {
    async function loadModel() {

      try {
        await tf.ready();

        console.log("loading model.json");
        const modelJson = require('./assets/model/model.json');
        console.log("successfully loaded model.json: " + modelJson)

        console.log("loading model weights")

        const modelWeights1 = require('./assets/model/group1-shard1of25.bin');
        const modelWeights2 = require('./assets/model/group1-shard2of25.bin');
        const modelWeights3 = require('./assets/model/group1-shard3of25.bin');
        const modelWeights4 = require('./assets/model/group1-shard4of25.bin');
        const modelWeights5 = require('./assets/model/group1-shard5of25.bin');
        const modelWeights6 = require('./assets/model/group1-shard6of25.bin');
        const modelWeights7 = require('./assets/model/group1-shard7of25.bin');
        const modelWeights8 = require('./assets/model/group1-shard8of25.bin');
        const modelWeights9 = require('./assets/model/group1-shard9of25.bin');
        const modelWeights10 = require('./assets/model/group1-shard10of25.bin');
        const modelWeights11 = require('./assets/model/group1-shard11of25.bin');
        const modelWeights12 = require('./assets/model/group1-shard12of25.bin');
        const modelWeights13 = require('./assets/model/group1-shard13of25.bin');
        const modelWeights14 = require('./assets/model/group1-shard14of25.bin');
        const modelWeights15 = require('./assets/model/group1-shard15of25.bin');
        const modelWeights16 = require('./assets/model/group1-shard16of25.bin');
        const modelWeights17 = require('./assets/model/group1-shard17of25.bin');
        const modelWeights18 = require('./assets/model/group1-shard18of25.bin');
        const modelWeights19 = require('./assets/model/group1-shard19of25.bin');
        const modelWeights20 = require('./assets/model/group1-shard20of25.bin');
        const modelWeights21 = require('./assets/model/group1-shard21of25.bin');
        const modelWeights22 = require('./assets/model/group1-shard22of25.bin');
        const modelWeights23 = require('./assets/model/group1-shard23of25.bin');
        const modelWeights24 = require('./assets/model/group1-shard24of25.bin');
        const modelWeights25 = require('./assets/model/group1-shard25of25.bin');

        

        const allModelWeights = [
          modelWeights1, modelWeights2, modelWeights3, modelWeights4, modelWeights5,
          modelWeights6, modelWeights7, modelWeights8, modelWeights9, modelWeights10,
          modelWeights11, modelWeights12, modelWeights13, modelWeights14, modelWeights15,
          modelWeights16, modelWeights17, modelWeights18, modelWeights19, modelWeights20,
          modelWeights21, modelWeights22, modelWeights23, modelWeights24, modelWeights25
        ];

        console.log("successfully loaded all model weights: " + allModelWeights)

        console.log("creating graph model")
        const loadedModel = await tf.loadGraphModel(bundleResourceIO(modelJson, allModelWeights));
        console.log("successfully created graph model")
        // setModel(loadedModel);

      } catch (err) {
        console.error(err);
      }
    }

    loadModel();
  }, []);

  useEffect(() => {
    async function classifyImage() {
      if (model) {
        const response = await fetch('https://upload.wikimedia.org/wikipedia/commons/a/a4/Striped_hyena_%28Hyaena_hyaena%29_-_cropped.jpg', {}, { isBinary: true });
        const imageData = await response.arrayBuffer();
        const imageTensor = tf.browser.fromPixels(imageData).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
        const prediction = await model.predict(imageTensor);
        setPredictions(prediction);
      }
    }

    classifyImage();
  }, [model]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Image Classification with ResNet50</Text>
      <Image
        source={{ uri: 'https://upload.wikimedia.org/wikipedia/commons/a/a4/Striped_hyena_%28Hyaena_hyaena%29_-_cropped.jpg' }}
        style={styles.image}
      />
      {predictions && (
        <Text style={styles.predictions}>
          Predictions: {JSON.stringify(predictions)}
        </Text>
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
