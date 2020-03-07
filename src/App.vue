<template>
  <div id="app">
    <!-- <div id="nav">
      <router-link to="/">Home</router-link> |
      <router-link to="/about">About</router-link>
    </div>
    <router-view /> -->
    test
  </div>
</template>

<script>
import Twitter from "twitter-lite";

const client = new Twitter({
  consumer_key: "WUDm8KuJMbTfvEWMo4YxDxJkA",
  consumer_secret: "2JJyKdylCY0bqWv8EFjoSGvKYsDz4wl7EceuSqbVt09bl5CSze"
});

export default {
  // mounted() {
  //   client
  //     .getRequestToken("http://localhost:8080/callback")
  //     .then(res =>
  //       console.log({
  //         reqTkn: res.oauth_token,
  //         reqTknSecret: res.oauth_token_secret
  //       })
  //     )
  //     .catch(console.error);
  // }
};
</script>
