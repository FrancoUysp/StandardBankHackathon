import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  ActivityIndicator,
  Linking,
  TouchableOpacity,
  Image,
} from "react-native";
import axios from "axios"; // You'll need to install this package
import { db } from "../../FirebaseConfig";
import { collection, getDocs } from "firebase/firestore";

const MunicipalityScreen = () => {
  const [potholes, setPotholes] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPotholes();
  }, []);

  const fetchPotholes = async () => {
    try {
      // Replace with your actual Firebase API endpoint
      // const response = await axios.get(
      //   "https://your-firebase-api-endpoint.com/potholes"
      // );
      const potholeCollection = collection(db, "potholes");
      const potholeSnapshot = await getDocs(potholeCollection);
      const potholeList = potholeSnapshot.docs.map((doc) => ({
        ...doc.data(),
        id: doc.id,
      }));
      setPotholes(potholeList);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching potholes:", error);
      setLoading(false);
    }
  };

  const renderPotholeItem = ({ item }) => {
    const googleMapsUrl = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(
      item.location
    )}`;

    return (
      <View style={styles.potholeItem}>
        <View style={{ alignItems: 'center' }}>
        <Image source={{ uri: item.image }} style={styles.potholeImage} />
        </View>
        <Text style={styles.potholeTitle}>Pothole ID: {item.id}</Text>
        <Text>Description: {item.description}</Text>
        <Text>Location: {item.location}</Text>
        <Text>
          Reported on: {item.reported_date.toDate().toLocaleDateString()}
        </Text>
        <Text>Status: {item.status}</Text>
        <Text>Bags: {item.bags}</Text>
        <TouchableOpacity onPress={() => Linking.openURL(googleMapsUrl)}>
          <Text style={styles.linkText}>View in Google Maps</Text>
        </TouchableOpacity>
      </View>
    );
  };

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#0000ff" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Reported Potholes</Text>
      <FlatList
        data={potholes}
        renderItem={renderPotholeItem}
        keyExtractor={(item) => item.id.toString()}
        ListEmptyComponent={
          <Text style={styles.emptyList}>No potholes reported yet.</Text>
        }
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: "#F5FCFF",
  },
  centered: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 16,
  },
  potholeItem: {
    backgroundColor: "white",
    padding: 16,
    marginBottom: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#ddd",
  },
  potholeTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 8,
  },
  potholeImage: {
    width: 100,
    height: 100,
    borderRadius: 50, // This makes the image circular
    marginBottom: 10,
  },
  emptyList: {
    textAlign: "center",
    marginTop: 50,
    fontSize: 18,
  },
  linkText: {
    color: "#1E90FF", // Blue color for the link
    marginTop: 8,
    fontSize: 16,
    textDecorationLine: "underline", // Underline the text to signify it's a link
  },
});

export default MunicipalityScreen;
