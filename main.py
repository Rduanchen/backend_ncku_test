import json
import sentry_sdk
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.middleware.cors import CORSMiddleware
import itertools
from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session, sessionmaker
from typing import List, Optional
import requests
from fastapi import APIRouter, HTTPException, Query, Depends, status, FastAPI
import os
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

from pydantic import AnyHttpUrl

from pydantic import BaseModel, Field, AnyHttpUrl
from sqlalchemy import (Column, ForeignKey, Integer, String, Table, Text,
                        create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

user_news_association_table = Table(
    "user_news_upvotes",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column(
        "news_articles_id", Integer, ForeignKey("news_articles.id"), primary_key=True
    ),
)

from pydantic import BaseModel


class NecessityPrice(BaseModel):
    類別: str
    編號: int
    產品名稱: str
    規格: str
    統計值: str
    時間起點: str
    時間終點: str


class NewsArticleSchema(BaseModel):
    id: int
    url: str
    title: str
    time: str
    content: str
    summary: str
    reason: str
    upvotes: int
    is_upvoted: bool

    class Config:
        from_attributes = True


class SearchNewsArticleSchema(BaseModel):
    id: int
    url: str
    title: str
    time: str
    content: str


class NewsSummarySchema(BaseModel):
    summary: str
    reason: str


class NewsSumaryRequestSchema(BaseModel):
    content: str


class PromptRequest(BaseModel):
    prompt: str


class TokenSchema(BaseModel):
    access_token: str
    token_type: str


class UserSchema(BaseModel):
    username: str

    class Config:
        from_attributes = True


class UserAuthSchema(BaseModel):
    username: str
    password: str


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    upvoted_news = relationship(
        "NewsArticle",
        secondary=user_news_association_table,
        back_populates="upvoted_by_users",
    )


class NewsArticle(Base):
    __tablename__ = "news_articles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    time = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    reason = Column(Text, nullable=False)
    upvoted_by_users = relationship(
        "User", secondary=user_news_association_table, back_populates="upvoted_news"
    )


engine = create_engine("sqlite:///news_database.db", echo=True)

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)


class Headline(BaseModel):
    title: str = Field(
        default=...,
        example="Title of the article",
        description="The title of the article"
    )
    url: AnyHttpUrl | str = Field(
        default=...,
        example="https://www.example.com",
        description="The URL of the article"
    )

class News(Headline):
    time: str = Field(
        default=...,
        example="2021-10-01T00:00:00",
        description="The time the article was published"
    )
    content: str = Field(
        default=...,
        example="Content of the article",
        description="The content of the article"
    )

class NewsWithSummary(News):
    summary: str = Field(
        default=...,
        example="Summary of the article",
        description="The summary of the article"
    )
    reason: str = Field(
        default=...,
        example="Reason of the article",
        description="The reason of the article"
    )

sentry_sdk.init(
    dsn="https://4001ffe917ccb261aa0e0c34026dc343@o4505702629834752.ingest.us.sentry.io/4507694792704000",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
)

app = FastAPI()
scheduler = BackgroundScheduler()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app.add_middleware(
    CORSMiddleware, # noqa
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")


class OpenAIClient:
    def __init__(self, _api_key: str):
        self.client = OpenAI(api_key=_api_key)

    def _generate_text(self, messages=None, model="gpt-3.5-turbo"):
        if not messages:
            messages = [
                {
                    "role": "system",
                    "content": "你是一個對話機器人，請回答以下問題。",
                }
            ]
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return completion.choices[0].message.content

    def evaluate_relevance(self, title):
        messages = [
            {
                "role": "system",
                "content": "你是一個關聯度評估機器人，請評估新聞標題是否與「民生用品的價格變化」相關，並給予'high'、'medium'、'low'評價。(僅需回答'high'、'medium'、'low'三個詞之一)",
            },
            {"role": "user", "content": f"{title}"},
        ]
        return openai_client._generate_text(messages=messages)

    def generate_summary(self, content):
        messages = [
            {
                "role": "system",
                "content": "你是一個新聞摘要生成機器人，請統整新聞中提及的影響及主要原因 (影響、原因各50個字，請以json格式回答 {'影響': '...', '原因': '...'})",
            },
            {"role": "user", "content": f"{content}"},
        ]
        return openai_client._generate_text(messages=messages)

    def extract_search_keywords(self, content):
        messages = [
            {
                "role": "system",
                "content": "你是一個關鍵字提取機器人，用戶將會輸入一段文字，表示其希望看見的新聞內容，請提取出用戶希望看見的關鍵字，請截取最重要的關鍵字即可，避免出現「新聞」、「資訊」等混淆搜尋引擎的字詞。(僅須回答關鍵字，若有多個關鍵字，請以空格分隔)",
            },
            {"role": "user", "content": f"{content}"},
        ]
        return openai_client._generate_text(messages=messages)


openai_client = OpenAIClient(_api_key=api_key)


from urllib.parse import quote
import requests
from requests import Response
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session


class UDNCrawler():
    CHANNEL_ID = 2

    def __init__(self, timeout = 5):
        self.news_website_url = "https://udn.com/api/more"
        self.timeout = timeout

    def startup(self, search_term: str) -> list[Headline]:
        headlines = []
        for p in range(1, 10):
            params = {
                "page": p,
                "id": f"search:{quote(search_term)}",
                "channelId": self.CHANNEL_ID,
                "type": "searchword",
            }
            response = self.perform_request(params=params)
            headline_list = self.parse_headlines(response) if response else []
            for headline in headline_list:
                headlines.append(headline)
        return headlines

    def get_headline(self, search_term, page):
        headlines = []
        page_range = range(*page) if isinstance(page, tuple) else [page]
        for p in page_range:
            params = {
                "page": p,
                "id": f"search:{quote(search_term)}",
                "channelId": self.CHANNEL_ID,
                "type": "searchword",
            }
            response = self.perform_request(params=params)
            headline_list = self.parse_headlines(response) if response else []
            for headline in headline_list:
                headlines.append(headline)
        return headlines

    def perform_request(self, url = None, params = None) -> Response:
        if not url:
            url = self.news_website_url

        if not params:
            params = {}

        try:
            response = requests.get(
                url, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            raise ConnectionError(f"Error fetching news: {e}") from e

    @staticmethod
    def parse_headlines(response: Response):
        data = response.json().get("lists", [])
        return [
            Headline(title=article["title"], url=article["titleLink"])
            for article in data
        ]

    def parse(self, url):
        response = self.perform_request(url=url)
        soup = BeautifulSoup(response.text, "html.parser")
        return self.extract_news(soup, url)

    @staticmethod
    def extract_news(soup: BeautifulSoup, url):
        title = soup.select_one("h1.article-content__title").text
        time = soup.select_one("time.article-content__time").text
        content = " ".join(
            p.text
            for p in soup.select("section.article-content__editor p")
            if p.text.strip()
        )
        return News(url=url, title=title, time=time, content=content, reason="", summary="")

    def save(self, news: NewsWithSummary, db: Session):
        existing_news = db.query(NewsArticle).filter_by(url=news.url).first()
        if not existing_news:
            new_article = NewsArticle(
                url=news.url, title=news.title, time=news.time, content=news.content, summary=news.summary, reason=news.reason
            )
            db.add(new_article)
            self.commit_changes(db)

    @staticmethod
    def commit_changes(db: Session):
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise RuntimeError(f"Error saving news to database: {e}") from e
        finally:
            db.close()



def fetch_news_task(is_init: bool = False):
    db = SessionLocal()
    # should change into simple factory pattern
    udn_crawler = UDNCrawler()
    if is_init:
        news_data = udn_crawler.startup("價格")
    else:
        news_data = udn_crawler.get_headline("價格", 1)
    for news in news_data:
        try:
            relevance = openai_client.evaluate_relevance(news.title)
            if relevance == "high":
                detailed_news = udn_crawler.parse(news.url)
                if detailed_news:
                    result = openai_client.generate_summary(detailed_news.content)
                    if result:
                        result = json.loads(result)
                        news_with_summary = NewsWithSummary(
                            title=detailed_news.title,
                            url=detailed_news.url,
                            time=detailed_news.time,
                            content=detailed_news.content,
                            summary=result.get("影響", ""),
                            reason=result.get("原因", "")
                        )
                        print(news_with_summary)
                        udn_crawler.save(news_with_summary, db)
                        print(f"Saved news {detailed_news.url}")
        except Exception as e:
            print(f"Error processing news {news.url}: {e}")


@app.on_event("startup")
def start_scheduler():
    db = SessionLocal()
    if db.query(NewsArticle).count() == 0:
        # should change into simple factory pattern
        fetch_news_task(is_init=True)
    db.close()
    scheduler.add_job(fetch_news_task, "interval", minutes=100)
    scheduler.start()


@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()


@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0 # noqa




# users api
app = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.environ.get("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/users/login")

def session_opener():
    session = Session(bind=engine)
    try:
        yield session
    finally:
        session.close()

def get_user(db: Session, name: str):
    return db.query(User).filter(User.username == name).first()


def verify(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(db: Session, username: str, pwd: str):
    user = get_user(db, username)
    if not user or not verify(pwd, user.hashed_password):
        return False
    return user


def authenticate_user_token(
    token: str = Depends(oauth2_scheme), db: Session = Depends(session_opener)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    print(to_encode)
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@app.post("/login", response_model=TokenSchema)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(session_opener)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.username)}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/register", response_model=UserSchema)
def create_user(user: UserAuthSchema, db: Session = Depends(session_opener)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = pwd_context.hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.get("/me", response_model=UserSchema)
def read_users_me(user: str = Depends(authenticate_user_token)):
    return {"username": user.username}


#news api
app = APIRouter()

udn_crawler = UDNCrawler()

_id_counter = itertools.count(start=1000000)

def get_article_upvote_details(article_id, uid, db):
    cnt = (
        db.query(user_news_association_table)
        .filter_by(news_articles_id=article_id)
        .count()
    )
    voted = False
    if uid:
        voted = (
            db.query(user_news_association_table)
            .filter_by(news_articles_id=article_id, user_id=uid)
            .first()
            is not None
        )
    return cnt, voted


@app.get("/news", response_model=list[NewsArticleSchema])
def read_news(db: Session = Depends(session_opener)):
    news = db.query(NewsArticle).order_by(NewsArticle.time.desc()).all()
    result = []
    for n in news:
        upvotes, upvoted = get_article_upvote_details(n.id, None, db)
        result.append(
            {**n.__dict__, "upvotes": upvotes, "is_upvoted": upvoted}
        )
    return result


@app.get(
    "/user_news",
    response_model=list[NewsArticleSchema],
    description="獲取包含user upvote資訊的新聞列表",
)
def read_user_news(
    db: Session = Depends(session_opener), u: User = Depends(authenticate_user_token)
):
    news = db.query(NewsArticle).order_by(NewsArticle.time.desc()).all()
    result = []
    for article in news:
        upvotes, upvoted = get_article_upvote_details(article.id, u.id, db)
        result.append(
            {
                **article.__dict__,
                "upvotes": upvotes,
                "is_upvoted": upvoted,
            }
        )
    return result


@app.post("/search_news", response_model=list[SearchNewsArticleSchema])
async def search_news(request: PromptRequest):
    prompt = request.prompt
    newss = []
    keywords = openai_client.extract_search_keywords(prompt)
    if not keywords:
        return []
    # should change into simple factory pattern
    news = udn_crawler.get_headline(keywords, page=1)
    for n in news:
        try:
            detailed_n = udn_crawler.parse(n.url)
            if detailed_n:
                detailed_news_dict = detailed_n.model_dump()
                detailed_news_dict["id"] = next(_id_counter)
                newss.append(detailed_news_dict)
        except Exception as e:
            print(f"Error processing news {n.url}: {e}")
    return sorted(newss, key=lambda x: x["time"], reverse=True)


@app.post("/news_summary", response_model=NewsSummarySchema)
async def news_summary(
    payload: NewsSumaryRequestSchema, user=Depends(authenticate_user_token)
):
    content = payload.content
    response = {}
    result = openai_client.generate_summary(content)
    if result:
        result = json.loads(result)
        response["summary"] = result["影響"]
        response["reason"] = result["原因"]
    return response

@app.post("/{id}/upvote")
def upvote_article(
    id: int,
    db: Session = Depends(session_opener),
    u: User = Depends(authenticate_user_token),
):
    if not u:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not exists(id, db):
        raise HTTPException(status_code=404, detail="News not found")

    message = toggle_upvote(id, u.id, db)
    return {"message": message}


def toggle_upvote(n_id: int, u_id: int, db: Session):
    existing_upvote = db.execute(
        select(user_news_association_table).where(
            user_news_association_table.c.news_articles_id == n_id,
            user_news_association_table.c.user_id == u_id,
        )
    ).scalar()

    if existing_upvote:
        delete_stmt = delete(user_news_association_table).where(
            user_news_association_table.c.news_articles_id == n_id,
            user_news_association_table.c.user_id == u_id,
        )
        db.execute(delete_stmt)
        db.commit()
        return "Upvote removed"
    else:
        insert_stmt = insert(user_news_association_table).values(
            news_articles_id=n_id, user_id=u_id
        )
        db.execute(insert_stmt)
        db.commit()
        return "Article upvoted"


def exists(news_id: int, db: Session):
    return db.query(NewsArticle).filter_by(id=news_id).first() is not None


app = APIRouter()

@app.get("/necessities-price", response_model=List[NecessityPrice])
def get_necessities_prices(
    category: Optional[str] = Query(None), commodity: Optional[str] = Query(None)
):
    response = requests.get(
        "https://opendata.ey.gov.tw/api/ConsumerProtection/NecessitiesPrice",
        params={"CategoryName": category, "Name": commodity},
    )
    response.raise_for_status()


    return response.json()

app.include_router(app, prefix="/v1")