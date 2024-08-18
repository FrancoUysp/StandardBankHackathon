// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAmft5YhdzKx1rLGwjvJnM3ltmZphJW_ic",
  authDomain: "patchperfect-98abd.firebaseapp.com",
  projectId: "patchperfect-98abd",
  storageBucket: "patchperfect-98abd.appspot.com",
  messagingSenderId: "870024655879",
  appId: "1:870024655879:web:27ad274eacdc2ef3e0e6a2",
  measurementId: "G-4E3F2W2JKP"
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
export const storage = getStorage(app);
