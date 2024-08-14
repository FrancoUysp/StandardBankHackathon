import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

const LoginScreen = ({ navigation }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.welcomeText}>Report a Pot Hole!</Text>

      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate("Main", { screen: "User" })}
      >
        <Text style={styles.buttonText}>Access User Page</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate("Main", { screen: "Municipality" })}
      >
        <Text style={styles.buttonText}>Access Municipality page</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  welcomeText: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 40,
    textAlign: "center",
  },
  button: {
    backgroundColor: "blue",
    padding: 15,
    borderRadius: 5,
    width: "80%",
    alignItems: "center",
    marginBottom: 20,
  },
  buttonText: {
    color: "white",
    fontSize: 18,
    fontWeight: "bold",
  },
});

export default LoginScreen;
