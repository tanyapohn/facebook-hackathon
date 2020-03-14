const Twitter = require('twitter-lite');
const fs = require('fs');

require('dotenv').config();

const fileName = 'tweets.txt';

/* Beginning of Twitter keys configurations */
const consumer_key = process.env.CONSUMER_KEY;
const consumer_secret = process.env.CONSUMER_SECRET;

const oauth_token = process.env.OAUTH_TOKEN;
const oauth_token_secret = process.env.OAUTH_TOKEN_SECRET;
const oauth_verifier = process.env.OAUTH_VERIFIER;
const oauth_callback_confirmed = true;

const access_token_key = process.env.ACCESS_TOKEN_KEY;
const access_token_secret = process.env.ACCESS_TOKEN_SECRET;
const user_id = process.env.USER_ID;
const screen_name = process.env.SCREEN_NAME;
/* End of Twitter keys configurations */

// Creates Twitter instance
const client = new Twitter({
  consumer_key,
  consumer_secret,
  access_token_key,
  access_token_secret,
});

function getMentionTimeline(since_id) {
  return client.get('statuses/mentions_timeline', {
    since_id,
  });
}

function appendFile(text) {
  fs.appendFile(fileName, text + '\n', function(err) {
    if (err) throw err;
    console.log(`${new Date().toISOString()}: Found new Tweet: ${text}`);
  });
}

function checkTweet({ text }) {
  const _text = text.replace(`@${screen_name} `, '');
}

function getLatestTweetId() {
  const files = readFile();
  return files[files.length - 1].id_str;
}

function readFile() {
  let _tweets = fs.readFileSync(fileName, 'utf8');
  _tweets = _tweets.split('\n');
  _tweets.pop();
  return _tweets.map(l => JSON.parse(l));
}

function replyToTweet(tweet) {
  checkTweet(tweet);
  client
    .post('statuses/update', {
      status: `@${tweet.author} Sir, your random number is ${Math.random()}`,
      in_reply_to_status_id: tweet.id_str,
    })
    .then(res => {
      console.log(
        `${new Date().toISOString()}: Replied to @${
          res.in_reply_to_screen_name
        }. Text: "${res.text}". (id: ${res.id_str})`
      );
    });
}

function save2File(res) {
  const mappedTweet = res.reverse().map(t => ({
    created_at: new Date(t.created_at),
    id_str: t.id_str,
    author: t.user.screen_name,
    text: t.text,
  }));

  mappedTweet.forEach(t => {
    appendFile(JSON.stringify(t));
    replyToTweet(t);
  });
}

function main() {
  const latestTweet = getLatestTweetId();
  console.log(`${new Date().toISOString()}: Fetching new Tweets`);
  getMentionTimeline(latestTweet).then(save2File);
}

function keepLoadingTweets() {
  setInterval(main, 10 /* seconds */ * 1000 /* ms */);
}

keepLoadingTweets();
