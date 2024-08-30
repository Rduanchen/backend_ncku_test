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
        "news_article_id", Integer, ForeignKey("news_articles.id"), primary_key=True
    ),
)


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

sentry_sdk.init(
    dsn="https://4001ffe917ccb261aa0e0c34026dc343@o4505702629834752.ingest.us.sentry.io/4507694792704000",
    traces_sample_rate=1.0,
    profiles_sample_rate=1.0,
)

app = FastAPI()
background_scheduler = BackgroundScheduler()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from urllib.parse import quote
from bs4 import BeautifulSoup


def add_news(news_data):
    """
    Add news to the database.
    :param news_data: news information
    :return:
    """
    session = Session()
    session.add(NewsArticle(
        url=news_data["url"],
        title=news_data["title"],
        time=news_data["time"],
        content=" ".join(news_data["content"]),  # Convert content list to string
        summary=news_data["summary"],
        reason=news_data["reason"],
    ))
    session.commit()
    session.close()


def get_news_data(search_term, is_initial=False):
    """
    Get news data.

    :param search_term:
    :param is_initial:
    :return:
    """
    all_news_data = []
    if is_initial:
        news_pages = []
        for page in range(1, 10):
            query_params = {
                "page": page,
                "id": f"search:{quote(search_term)}",
                "channelId": 2,
                "type": "searchword",
            }
            response = requests.get("https://udn.com/api/more", params=query_params)
            news_pages.append(response.json()["lists"])

        for news_list in news_pages:
            all_news_data.append(news_list)
    else:
        query_params = {
            "page": 1,
            "id": f"search:{quote(search_term)}",
            "channelId": 2,
            "type": "searchword",
        }
        response = requests.get("https://udn.com/api/more", params=query_params)

        all_news_data = response.json()["lists"]
    return all_news_data

def fetch_news(is_initial=False):
    """
    Fetch news information.

    :param is_initial:
    :return:
    """
    news_data = get_news_data("價格", is_initial=is_initial)
    for news in news_data:
        system_message = {
            "role": "system",
            "content": "你是一個關聯度評估機器人，請評估新聞標題是否與「民生用品的價格變化」相關，並給予'high'、'medium'、'low'評價。(僅需回答'high'、'medium'、'low'三個詞之一)",
        }
        user_message = {"role": "user", "content": f"{news['title']}"}

        ai_response = OpenAI(api_key="xxx").chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_message, user_message],
        )
        relevance = ai_response.choices[0].message.content
        if relevance == "high":
            response = requests.get(news["titleLink"])
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.find("h1", class_="article-content__title").text
            time = soup.find("time", class_="article-content__time").text
            content_section = soup.find("section", class_="article-content__editor")

            paragraphs = [
                p.text
                for p in content_section.find_all("p")
                if p.text.strip() != "" and "▪" not in p.text
            ]
            detailed_news = {
                "url": news["titleLink"],
                "title": title,
                "time": time,
                "content": paragraphs,
            }
            system_message_summary = {
                "role": "system",
                "content": "你是一個新聞摘要生成機器人，請統整新聞中提及的影響及主要原因 (影響、原因各50個字，請以json格式回答 {'影響': '...', '原因': '...'})",
            }
            user_message_summary = {"role": "user", "content": " ".join(detailed_news["content"])}

            completion = OpenAI(api_key="xxx").chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[system_message_summary, user_message_summary],
            )
            result = json.loads(completion.choices[0].message.content)
            detailed_news["summary"] = result["影響"]
            detailed_news["reason"] = result["原因"]
            add_news(detailed_news)


@app.on_event("startup")
def start_scheduler():
    db_session = SessionLocal()
    if db_session.query(NewsArticle).count() == 0:
        fetch_news()
    db_session.close()
    background_scheduler.add_job(fetch_news, "interval", minutes=100)
    background_scheduler.start()


@app.on_event("shutdown")
def shutdown_scheduler():
    background_scheduler.shutdown()


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/users/login")


def open_session():
    session = Session(bind=engine)
    try:
        yield session
    finally:
        session.close()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def check_user_password(db_session, username, password):
    user = db_session.query(User).filter(User.username == username).first()
    if not verify_password(password, user.hashed_password):
        return False
    return user


def authenticate_user(token=Depends(oauth2_scheme), db_session=Depends(open_session)):
    payload = jwt.decode(token, '1892dhianiandowqd0n', algorithms=["HS256"])
    return db_session.query(User).filter(User.username == payload.get("sub")).first()


def create_access_token(data, expires_delta=None):
    """Create access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, '1892dhianiandowqd0n', algorithm="HS256")
    return encoded_jwt


@app.post("/api/v1/users/login")
async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(), db_session: Session = Depends(open_session)
):
    """Login."""
    user = check_user_password(db_session, form_data.username, form_data.password)
    access_token = create_access_token(
        data={"sub": str(user.username)}, expires_delta=timedelta(minutes=30)
    )
    return {"access_token": access_token, "token_type": "bearer"}

class UserAuthSchema(BaseModel):
    username: str
    password: str

@app.post("/api/v1/users/register")
def create_user(user: UserAuthSchema, db_session: Session = Depends(open_session)):
    """Create user."""
    hashed_password = pwd_context.hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    db_session.add(db_user)
    db_session.commit()
    db_session.refresh(db_user)
    return db_user


@app.get("/api/v1/users/me")
def read_users_me(user=Depends(authenticate_user)):
    return {"username": user.username}


id_counter = itertools.count(start=1000000)


def get_article_upvote_details(article_id, user_id, db_session):
    count = (
        db_session.query(user_news_association_table)
        .filter_by(news_article_id=article_id)
        .count()
    )
    is_voted = False
    if user_id:
        is_voted = (
                db_session.query(user_news_association_table)
                .filter_by(news_article_id=article_id, user_id=user_id)
                .first()
                is not None
        )
    return count, is_voted


@app.get("/api/v1/news/news")
def read_news(db_session=Depends(open_session)):
    """
    Read news.

    :param db_session:
    :return:
    """
    news = db_session.query(NewsArticle).order_by(NewsArticle.time.desc()).all()
    result = []
    for article in news:
        upvotes, is_upvoted = get_article_upvote_details(article.id, None, db_session)
        result.append(
            {**article.__dict__, "upvotes": upvotes, "is_upvoted": is_upvoted}
        )
    return result


@app.get("/api/v1/news/user_news")
def read_user_news(
        db_session=Depends(open_session),
        user=Depends(authenticate_user)
):
    """
    Read user news.

    :param db_session:
    :param user:
    :return:
    """
    news = db_session.query(NewsArticle).order_by(NewsArticle.time.desc()).all()
    result = []
    for article in news:
        upvotes, is_upvoted = get_article_upvote_details(article.id, user.id, db_session)
        result.append(
            {
                **article.__dict__,
                "upvotes": upvotes,
                "is_upvoted": is_upvoted,
            }
        )
    return result

class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/v1/news/search_news")
async def search_news(request: PromptRequest):
    prompt = request.prompt
    news_list = []
    system_message = {
        "role": "system",
        "content": "你是一個關鍵字提取機器人，用戶將會輸入一段文字，表示其希望看見的新聞內容，請提取出用戶希望看見的關鍵字，請截取最重要的關鍵字即可，避免出現「新聞」、「資訊」等混淆搜尋引擎的字詞。(僅須回答關鍵字，若有多個關鍵字，請以空格分隔)",
    }
    user_message = {"role": "user", "content": f"{prompt}"}

    completion = OpenAI(api_key="xxx").chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[system_message, user_message],
    )
    keywords = completion.choices[0].message.content
    news_items = get_news_data(keywords, is_initial=False)
    for news in news_items:
        try:
            response = requests.get(news["titleLink"])
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.find("h1", class_="article-content__title").text
            time = soup.find("time", class_="article-content__time").text
            content_section = soup.find("section", class_="article-content__editor")

            paragraphs = [
                p.text
                for p in content_section.find_all("p")
                if p.text.strip() != "" and "▪" not in p.text
            ]
            detailed_news = {
                "url": news["titleLink"],
                "title": title,
                "time": time,
                "content": paragraphs,
            }
            detailed_news["content"] = " ".join(detailed_news["content"])
            detailed_news["id"] = next(id_counter)
            news_list.append(detailed_news)
        except Exception as e:
            print(e)
    return sorted(news_list, key=lambda x: x["time"], reverse=True)

class NewsSummaryRequestSchema(BaseModel):
    content: str

@app.post("/api/v1/news/news_summary")
async def news_summary(
        payload: NewsSummaryRequestSchema, user=Depends(authenticate_user)
):
    response = {}
    system_message = {
        "role": "system",
        "content": "你是一個新聞摘要生成機器人，請統整新聞中提及的影響及主要原因 (影響、原因各50個字，請以json格式回答 {'影響': '...', '原因': '...'})",
    }
    user_message = {"role": "user", "content": f"{payload.content}"}

    completion = OpenAI(api_key="xxx").chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[system_message, user_message],
    )
    result = completion.choices[0].message.content
    if result:
        result = json.loads(result)
        response["summary"] = result["影響"]
        response["reason"] = result["原因"]
    return response


@app.post("/api/v1/news/{news_id}/upvote")
def upvote_article(
        news_id,
        db_session=Depends(open_session),
        user=Depends(authenticate_user),
):
    message = toggle_upvote(news_id, user.id, db_session)
    return {"message": message}


def toggle_upvote(news_id, user_id, db_session):
    existing_upvote = db_session.execute(
        select(user_news_association_table).where(
            user_news_association_table.c.news_article_id == news_id,
            user_news_association_table.c.user_id == user_id,
        )
    ).scalar()

    if existing_upvote:
        delete_stmt = delete(user_news_association_table).where(
            user_news_association_table.c.news_article_id == news_id,
            user_news_association_table.c.user_id == user_id,
        )
        db_session.execute(delete_stmt)
        db_session.commit()
        return "Upvote removed"
    else:
        insert_stmt = insert(user_news_association_table).values(
            news_article_id=news_id, user_id=user_id
        )
        db_session.execute(insert_stmt)
        db_session.commit()
        return "Article upvoted"


def news_exists(news_id, db_session: Session):
    return db_session.query(NewsArticle).filter_by(id=news_id).first() is not None


@app.get("/api/v1/prices/necessities-price")
def get_necessities_prices(
        category=Query(None), commodity=Query(None)
):
    return requests.get(
        "https://opendata.ey.gov.tw/api/ConsumerProtection/NecessitiesPrice",
        params={"CategoryName": category, "Name": commodity},
    ).json()
