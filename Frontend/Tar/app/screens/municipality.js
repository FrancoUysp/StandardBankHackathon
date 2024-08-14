import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  ActivityIndicator,
} from "react-native";
import axios from "axios"; // You'll need to install this package

const MunicipalityScreen = () => {
  const [potholes, setPotholes] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPotholes();
  }, []);

  const fetchPotholes = async () => {
    try {
      // Replace with your actual Firebase API endpoint
      const response = await axios.get(
        "https://your-firebase-api-endpoint.com/potholes"
      );
      setPotholes(response.data);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching potholes:", error);
      setLoading(false);
    }
  };

  const renderPotholeItem = ({ item }) => (
    <View style={styles.potholeItem}>
      <Text style={styles.potholeTitle}>Pothole ID: {item.id}</Text>
      <Text>Location: {item.location}</Text>
      <Text>
        Reported on: {new Date(item.reportedDate).toLocaleDateString()}
      </Text>
      <Text>Status: {item.status}</Text>
    </View>
  );

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
  emptyList: {
    textAlign: "center",
    marginTop: 50,
    fontSize: 18,
  },
});

export default MunicipalityScreen;
