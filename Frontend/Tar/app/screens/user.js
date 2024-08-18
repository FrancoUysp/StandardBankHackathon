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
import { db, storage } from "../../FirebaseConfig";
import { addDoc, collection } from "firebase/firestore";
import { getDownloadURL, ref, uploadBytes } from "firebase/storage"
import { getFunctions } from "firebase/functions";

export default function App() {
  const [cameraPermission, setCameraPermission] = useState(null);
  const [galleryPermission, setGalleryPermission] = useState(null);
  const [locationPermission, setLocationPermission] = useState(null);
  const [cameraRef, setCameraRef] = useState(null);
  const [image, setImage] = useState(null);
  const [description, setDescription] = useState("");
  const [location, setLocation] = useState(null);
  const [submitted, setSubmitted] = useState(false);

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

  const uploadImage = async () => {
    const response = await fetch(image);
    const blob = await response.blob();
    const storageRef = ref(storage, `potholes/${Date.now()}`);
    await uploadBytes(storageRef, blob);
    return getDownloadURL(storageRef);
  };

  const submitReport = async () => {
    const imageUrl = await uploadImage();
    const potholeCollection = collection(db, "potholes");
    docref = await addDoc(potholeCollection, {
      image: imageUrl,
      description: description,
      location: "Dagbreek Manskoshuis, Stellenbosch, 7600", // Replace this when location works
      reported_date: new Date(),
      status: "Reported",
    });

    // After submission, show the thank-you message and clear the screen
    setSubmitted(true);
    setImage(null);
    setDescription("");

    // Reset the form after a short delay (e.g., 3 seconds)
    setTimeout(() => {
      setSubmitted(false);
    }, 3000);
  };

  return (
    <View style={styles.container}>
      {submitted ? (
        <Text style={styles.thankYouText}>Thank you for submitting!</Text>
      ) : (
        <>
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
          <TouchableOpacity style={styles.submitButton} onPress={submitReport}>
            <Text style={styles.submitButtonText}>Submit</Text>
          </TouchableOpacity>
        </>
      )}
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
  thankYouText: {
    fontSize: 24,
    color: "#28a745",
    fontWeight: "bold",
  },
});
