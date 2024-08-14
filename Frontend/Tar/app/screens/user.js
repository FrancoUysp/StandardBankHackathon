import React, { useState } from "react";
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  TextInput,
} from "react-native";
import { Camera } from "expo-camera";
import * as ImagePicker from "expo-image-picker";
import * as Location from "expo-location";
import { db } from "../../FirebaseConfig";

export default function App() {
  const [cameraPermission, setCameraPermission] = useState(null);
  const [galleryPermission, setGalleryPermission] = useState(null);
  const [locationPermission, setLocationPermission] = useState(null);
  const [cameraRef, setCameraRef] = useState(null);
  const [image, setImage] = useState(null);
  const [description, setDescription] = useState("");
  const [location, setLocation] = useState(null);

  const requestPermissions = async () => {
    const cameraStatus = await Camera.requestPermissionsAsync();
    setCameraPermission(cameraStatus.status === "granted");

    const galleryStatus =
      await ImagePicker.requestMediaLibraryPermissionsAsync();
    setGalleryPermission(galleryStatus.status === "granted");

    const locationStatus = await Location.requestForegroundPermissionsAsync();
    setLocationPermission(locationStatus.status === "granted");
  };

  const takePhoto = async () => {
    if (cameraRef) {
      const photo = await cameraRef.takePictureAsync();
      setImage(photo.uri);
      const currentLocation = await Location.getCurrentPositionAsync({});
      setLocation(currentLocation);
    }
  };

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      const currentLocation = await Location.getCurrentPositionAsync({});
      setLocation(currentLocation);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Report a Pothole</Text>
      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.button} onPress={takePhoto}>
          <Text style={styles.buttonText}>Take a Photo</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.button} onPress={pickImage}>
          <Text style={styles.buttonText}>Upload from Gallery</Text>
        </TouchableOpacity>
      </View>
      {image && <Image source={{ uri: image }} style={styles.image} />}
      <TextInput
        style={styles.input}
        placeholder="Enter a description"
        onChangeText={setDescription}
        value={description}
      />
      <TouchableOpacity
        style={styles.submitButton}
        onPress={() => console.log({ image, description, location })}
      >
        <Text style={styles.submitButtonText}>Submit</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    fontSize: 24,
    marginBottom: 20,
  },
  buttonContainer: {
    width: "80%",
    alignItems: "center",
  },
  button: {
    backgroundColor: "#007AFF",
    padding: 15,
    borderRadius: 10,
    marginVertical: 10,
    width: "100%",
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontSize: 18,
  },
  image: {
    width: 200,
    height: 200,
    margin: 20,
  },
  input: {
    height: 40,
    borderColor: "gray",
    borderWidth: 1,
    width: "80%",
    marginBottom: 20,
    padding: 10,
    borderRadius: 5,
  },
  submitButton: {
    backgroundColor: "#28a745",
    padding: 15,
    borderRadius: 10,
    width: "80%",
    alignItems: "center",
  },
  submitButtonText: {
    color: "#fff",
    fontSize: 18,
  },
});
